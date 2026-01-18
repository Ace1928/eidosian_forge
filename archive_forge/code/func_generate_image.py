import sys
import os
import os.path
import tempfile
import zipfile
from xml.dom import minidom
import time
import re
import copy
import itertools
import docutils
from docutils import frontend, nodes, utils, writers, languages
from docutils.readers import standalone
from docutils.transforms import references
def generate_image(self, node, source, destination, current_element, frame_attrs=None):
    width, height = self.get_image_scaled_width_height(node, source)
    self.image_style_count += 1
    style_name = 'rstframestyle%d' % self.image_style_count
    attrib = {'style:name': style_name, 'style:family': 'graphic', 'style:parent-style-name': self.rststyle('image')}
    el1 = SubElement(self.automatic_styles, 'style:style', attrib=attrib, nsdict=SNSD)
    halign = None
    valign = None
    if 'align' in node.attributes:
        align = node.attributes['align'].split()
        for val in align:
            if val in ('left', 'center', 'right'):
                halign = val
            elif val in ('top', 'middle', 'bottom'):
                valign = val
    if frame_attrs is None:
        attrib = {'style:vertical-pos': 'top', 'style:vertical-rel': 'paragraph', 'style:horizontal-rel': 'paragraph', 'style:mirror': 'none', 'fo:clip': 'rect(0cm 0cm 0cm 0cm)', 'draw:luminance': '0%', 'draw:contrast': '0%', 'draw:red': '0%', 'draw:green': '0%', 'draw:blue': '0%', 'draw:gamma': '100%', 'draw:color-inversion': 'false', 'draw:image-opacity': '100%', 'draw:color-mode': 'standard'}
    else:
        attrib = frame_attrs
    if halign is not None:
        attrib['style:horizontal-pos'] = halign
    if valign is not None:
        attrib['style:vertical-pos'] = valign
    wrap = False
    classes = node.attributes.get('classes')
    if classes and 'wrap' in classes:
        wrap = True
    if wrap:
        attrib['style:wrap'] = 'dynamic'
    else:
        attrib['style:wrap'] = 'none'
    if self.is_in_table(node):
        attrib['style:wrap'] = 'none'
    SubElement(el1, 'style:graphic-properties', attrib=attrib, nsdict=SNSD)
    draw_name = 'graphics%d' % next(IMAGE_NAME_COUNTER)
    attrib = {'draw:style-name': style_name, 'draw:name': draw_name, 'draw:z-index': '1'}
    if isinstance(node.parent, nodes.TextElement):
        attrib['text:anchor-type'] = 'as-char'
    else:
        attrib['text:anchor-type'] = 'paragraph'
    attrib['svg:width'] = width
    attrib['svg:height'] = height
    el1 = SubElement(current_element, 'draw:frame', attrib=attrib)
    SubElement(el1, 'draw:image', attrib={'xlink:href': '%s' % (destination,), 'xlink:type': 'simple', 'xlink:show': 'embed', 'xlink:actuate': 'onLoad'})
    return (el1, width)