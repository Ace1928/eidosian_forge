from __future__ import division         # use "true" division instead of integer division in Python 2 (see PEP 238)
from __future__ import print_function   # use print() as a function in Python 2 (see PEP 3105)
from __future__ import absolute_import  # use absolute imports by default in Python 2 (see PEP 328)
import math
import optparse
import os
import re
import sys
import time
import xml.dom.minidom
from xml.dom import Node, NotFoundErr
from collections import namedtuple, defaultdict
from decimal import Context, Decimal, InvalidOperation, getcontext
import six
from six.moves import range, urllib
from scour.svg_regex import svg_parser
from scour.svg_transform import svg_transform_parser
from scour.yocto_css import parseCssString
from scour import __version__
def scourString(in_string, options=None):
    options = sanitizeOptions(options)
    if options.cdigits < 0:
        options.cdigits = options.digits
    global scouringContext
    global scouringContextC
    scouringContext = Context(prec=options.digits)
    scouringContextC = Context(prec=options.cdigits)
    global _num_elements_removed
    global _num_attributes_removed
    global _num_ids_removed
    global _num_comments_removed
    global _num_style_properties_fixed
    global _num_rasters_embedded
    global _num_path_segments_removed
    global _num_points_removed_from_polygon
    global _num_bytes_saved_in_path_data
    global _num_bytes_saved_in_colors
    global _num_bytes_saved_in_comments
    global _num_bytes_saved_in_ids
    global _num_bytes_saved_in_lengths
    global _num_bytes_saved_in_transforms
    _num_elements_removed = 0
    _num_attributes_removed = 0
    _num_ids_removed = 0
    _num_comments_removed = 0
    _num_style_properties_fixed = 0
    _num_rasters_embedded = 0
    _num_path_segments_removed = 0
    _num_points_removed_from_polygon = 0
    _num_bytes_saved_in_path_data = 0
    _num_bytes_saved_in_colors = 0
    _num_bytes_saved_in_comments = 0
    _num_bytes_saved_in_ids = 0
    _num_bytes_saved_in_lengths = 0
    _num_bytes_saved_in_transforms = 0
    doc = xml.dom.minidom.parseString(in_string)
    cnt_flowText_el = len(doc.getElementsByTagName('flowRoot'))
    if cnt_flowText_el:
        errmsg = "SVG input document uses {} flow text elements, which won't render on browsers!".format(cnt_flowText_el)
        if options.error_on_flowtext:
            raise Exception(errmsg)
        else:
            print('WARNING: {}'.format(errmsg), file=sys.stderr)
    removeDescriptiveElements(doc, options)
    if options.keep_editor_data is False:
        _num_elements_removed += removeNamespacedElements(doc.documentElement, unwanted_ns)
        _num_attributes_removed += removeNamespacedAttributes(doc.documentElement, unwanted_ns)
        xmlnsDeclsToRemove = []
        attrList = doc.documentElement.attributes
        for index in range(attrList.length):
            if attrList.item(index).nodeValue in unwanted_ns:
                xmlnsDeclsToRemove.append(attrList.item(index).nodeName)
        for attr in xmlnsDeclsToRemove:
            doc.documentElement.removeAttribute(attr)
            _num_attributes_removed += 1
    if doc.documentElement.getAttribute('xmlns') != 'http://www.w3.org/2000/svg':
        doc.documentElement.setAttribute('xmlns', 'http://www.w3.org/2000/svg')

    def xmlnsUnused(prefix, namespace):
        if doc.getElementsByTagNameNS(namespace, '*'):
            return False
        else:
            for element in doc.getElementsByTagName('*'):
                for attribute in element.attributes.values():
                    if attribute.name.startswith(prefix):
                        return False
        return True
    attrList = doc.documentElement.attributes
    xmlnsDeclsToRemove = []
    redundantPrefixes = []
    for i in range(attrList.length):
        attr = attrList.item(i)
        name = attr.nodeName
        val = attr.nodeValue
        if name[0:6] == 'xmlns:':
            if val == 'http://www.w3.org/2000/svg':
                redundantPrefixes.append(name[6:])
                xmlnsDeclsToRemove.append(name)
            elif xmlnsUnused(name[6:], val):
                xmlnsDeclsToRemove.append(name)
    for attrName in xmlnsDeclsToRemove:
        doc.documentElement.removeAttribute(attrName)
        _num_attributes_removed += 1
    for prefix in redundantPrefixes:
        remapNamespacePrefix(doc.documentElement, prefix, '')
    if options.strip_comments:
        _num_comments_removed = removeComments(doc)
    if options.strip_xml_space_attribute and doc.documentElement.hasAttribute('xml:space'):
        doc.documentElement.removeAttribute('xml:space')
        _num_attributes_removed += 1
    _num_style_properties_fixed = repairStyle(doc.documentElement, options)
    if options.simple_colors:
        _num_bytes_saved_in_colors = convertColors(doc.documentElement)
    while removeUnreferencedElements(doc, options.keep_defs) > 0:
        pass
    for tag in ['defs', 'title', 'desc', 'metadata', 'g']:
        for elem in doc.documentElement.getElementsByTagName(tag):
            removeElem = not elem.hasChildNodes()
            if removeElem is False:
                for child in elem.childNodes:
                    if child.nodeType in [Node.ELEMENT_NODE, Node.CDATA_SECTION_NODE, Node.COMMENT_NODE]:
                        break
                    elif child.nodeType == Node.TEXT_NODE and (not child.nodeValue.isspace()):
                        break
                else:
                    removeElem = True
            if removeElem:
                elem.parentNode.removeChild(elem)
                _num_elements_removed += 1
    if options.strip_ids:
        referencedIDs = findReferencedElements(doc.documentElement)
        identifiedElements = unprotected_ids(doc, options)
        removeUnreferencedIDs(referencedIDs, identifiedElements)
    while removeDuplicateGradientStops(doc) > 0:
        pass
    while collapseSinglyReferencedGradients(doc) > 0:
        pass
    _num_elements_removed += removeDuplicateGradients(doc)
    if options.group_collapse:
        _num_elements_removed += mergeSiblingGroupsWithCommonAttributes(doc.documentElement)
    if options.group_create:
        createGroupsForCommonAttributes(doc.documentElement)
    referencedIds = findReferencedElements(doc.documentElement)
    for child in doc.documentElement.childNodes:
        _num_attributes_removed += moveCommonAttributesToParentGroup(child, referencedIds)
    _num_attributes_removed += removeUnusedAttributesOnParent(doc.documentElement)
    if options.group_collapse:
        while removeNestedGroups(doc.documentElement) > 0:
            pass
    for polygon in doc.documentElement.getElementsByTagName('polygon'):
        cleanPolygon(polygon, options)
    for polyline in doc.documentElement.getElementsByTagName('polyline'):
        cleanPolyline(polyline, options)
    for elem in doc.documentElement.getElementsByTagName('path'):
        if elem.getAttribute('d') == '':
            elem.parentNode.removeChild(elem)
        else:
            cleanPath(elem, options)
    if options.shorten_ids:
        _num_bytes_saved_in_ids += shortenIDs(doc, options.shorten_ids_prefix, options)
    for type in ['svg', 'image', 'rect', 'circle', 'ellipse', 'line', 'linearGradient', 'radialGradient', 'stop', 'filter']:
        for elem in doc.getElementsByTagName(type):
            for attr in ['x', 'y', 'width', 'height', 'cx', 'cy', 'r', 'rx', 'ry', 'x1', 'y1', 'x2', 'y2', 'fx', 'fy', 'offset']:
                if elem.getAttribute(attr) != '':
                    elem.setAttribute(attr, scourLength(elem.getAttribute(attr)))
    viewBox = doc.documentElement.getAttribute('viewBox')
    if viewBox:
        lengths = RE_COMMA_WSP.split(viewBox)
        lengths = [scourUnitlessLength(length) for length in lengths]
        doc.documentElement.setAttribute('viewBox', ' '.join(lengths))
    _num_bytes_saved_in_lengths = reducePrecision(doc.documentElement)
    _num_attributes_removed += removeDefaultAttributeValues(doc.documentElement, options)
    _num_bytes_saved_in_transforms = optimizeTransforms(doc.documentElement, options)
    if options.embed_rasters:
        for elem in doc.documentElement.getElementsByTagName('image'):
            embedRasters(elem, options)
    if options.enable_viewboxing:
        properlySizeDoc(doc.documentElement, options)
    out_string = serializeXML(doc.documentElement, options) + '\n'
    if options.strip_xml_prolog is False:
        total_output = '<?xml version="1.0" encoding="UTF-8"'
        if doc.standalone:
            total_output += ' standalone="yes"'
        total_output += '?>\n'
    else:
        total_output = ''
    for child in doc.childNodes:
        if child.nodeType == Node.ELEMENT_NODE:
            total_output += out_string
        else:
            total_output += child.toxml() + '\n'
    return total_output