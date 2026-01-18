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
def generate_figure(self, node, source, destination, current_element):
    caption = None
    width, height = self.get_image_scaled_width_height(node, source)
    for node1 in node.parent.children:
        if node1.tagname == 'caption':
            caption = node1.astext()
    self.image_style_count += 1
    if caption is not None:
        attrib = {'style:class': 'extra', 'style:family': 'paragraph', 'style:name': 'Caption', 'style:parent-style-name': 'Standard'}
        el1 = SubElement(self.automatic_styles, 'style:style', attrib=attrib, nsdict=SNSD)
        attrib = {'fo:margin-bottom': '0.0835in', 'fo:margin-top': '0.0835in', 'text:line-number': '0', 'text:number-lines': 'false'}
        SubElement(el1, 'style:paragraph-properties', attrib=attrib, nsdict=SNSD)
        attrib = {'fo:font-size': '12pt', 'fo:font-style': 'italic', 'style:font-name': 'Times', 'style:font-name-complex': 'Lucidasans1', 'style:font-size-asian': '12pt', 'style:font-size-complex': '12pt', 'style:font-style-asian': 'italic', 'style:font-style-complex': 'italic'}
        SubElement(el1, 'style:text-properties', attrib=attrib, nsdict=SNSD)
    style_name = 'rstframestyle%d' % self.image_style_count
    draw_name = 'graphics%d' % next(IMAGE_NAME_COUNTER)
    attrib = {'style:name': style_name, 'style:family': 'graphic', 'style:parent-style-name': self.rststyle('figureframe')}
    el1 = SubElement(self.automatic_styles, 'style:style', attrib=attrib, nsdict=SNSD)
    attrib = {}
    wrap = False
    classes = node.parent.attributes.get('classes')
    if classes and 'wrap' in classes:
        wrap = True
    if wrap:
        attrib['style:wrap'] = 'dynamic'
    else:
        attrib['style:wrap'] = 'none'
    SubElement(el1, 'style:graphic-properties', attrib=attrib, nsdict=SNSD)
    attrib = {'draw:style-name': style_name, 'draw:name': draw_name, 'text:anchor-type': 'paragraph', 'draw:z-index': '0'}
    attrib['svg:width'] = width
    el3 = SubElement(current_element, 'draw:frame', attrib=attrib)
    attrib = {}
    el4 = SubElement(el3, 'draw:text-box', attrib=attrib)
    attrib = {'text:style-name': self.rststyle('caption')}
    el5 = SubElement(el4, 'text:p', attrib=attrib)
    return (el3, el4, el5, caption)