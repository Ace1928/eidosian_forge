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
def make_field_element(self, parent, text, style_name, automatic_styles):
    if text == 'p':
        el1 = SubElement(parent, 'text:page-number', attrib={'text:select-page': 'current'})
    elif text == 'P':
        el1 = SubElement(parent, 'text:page-count', attrib={})
    elif text == 't1':
        self.style_index += 1
        el1 = SubElement(parent, 'text:time', attrib={'text:style-name': self.rststyle(style_name), 'text:fixed': 'true', 'style:data-style-name': 'rst-time-style-%d' % self.style_index})
        el2 = SubElement(automatic_styles, 'number:time-style', attrib={'style:name': 'rst-time-style-%d' % self.style_index, 'xmlns:number': SNSD['number'], 'xmlns:style': SNSD['style']})
        el3 = SubElement(el2, 'number:hours', attrib={'number:style': 'long'})
        el3 = SubElement(el2, 'number:text')
        el3.text = ':'
        el3 = SubElement(el2, 'number:minutes', attrib={'number:style': 'long'})
    elif text == 't2':
        self.style_index += 1
        el1 = SubElement(parent, 'text:time', attrib={'text:style-name': self.rststyle(style_name), 'text:fixed': 'true', 'style:data-style-name': 'rst-time-style-%d' % self.style_index})
        el2 = SubElement(automatic_styles, 'number:time-style', attrib={'style:name': 'rst-time-style-%d' % self.style_index, 'xmlns:number': SNSD['number'], 'xmlns:style': SNSD['style']})
        el3 = SubElement(el2, 'number:hours', attrib={'number:style': 'long'})
        el3 = SubElement(el2, 'number:text')
        el3.text = ':'
        el3 = SubElement(el2, 'number:minutes', attrib={'number:style': 'long'})
        el3 = SubElement(el2, 'number:text')
        el3.text = ':'
        el3 = SubElement(el2, 'number:seconds', attrib={'number:style': 'long'})
    elif text == 't3':
        self.style_index += 1
        el1 = SubElement(parent, 'text:time', attrib={'text:style-name': self.rststyle(style_name), 'text:fixed': 'true', 'style:data-style-name': 'rst-time-style-%d' % self.style_index})
        el2 = SubElement(automatic_styles, 'number:time-style', attrib={'style:name': 'rst-time-style-%d' % self.style_index, 'xmlns:number': SNSD['number'], 'xmlns:style': SNSD['style']})
        el3 = SubElement(el2, 'number:hours', attrib={'number:style': 'long'})
        el3 = SubElement(el2, 'number:text')
        el3.text = ':'
        el3 = SubElement(el2, 'number:minutes', attrib={'number:style': 'long'})
        el3 = SubElement(el2, 'number:text')
        el3.text = ' '
        el3 = SubElement(el2, 'number:am-pm')
    elif text == 't4':
        self.style_index += 1
        el1 = SubElement(parent, 'text:time', attrib={'text:style-name': self.rststyle(style_name), 'text:fixed': 'true', 'style:data-style-name': 'rst-time-style-%d' % self.style_index})
        el2 = SubElement(automatic_styles, 'number:time-style', attrib={'style:name': 'rst-time-style-%d' % self.style_index, 'xmlns:number': SNSD['number'], 'xmlns:style': SNSD['style']})
        el3 = SubElement(el2, 'number:hours', attrib={'number:style': 'long'})
        el3 = SubElement(el2, 'number:text')
        el3.text = ':'
        el3 = SubElement(el2, 'number:minutes', attrib={'number:style': 'long'})
        el3 = SubElement(el2, 'number:text')
        el3.text = ':'
        el3 = SubElement(el2, 'number:seconds', attrib={'number:style': 'long'})
        el3 = SubElement(el2, 'number:text')
        el3.text = ' '
        el3 = SubElement(el2, 'number:am-pm')
    elif text == 'd1':
        self.style_index += 1
        el1 = SubElement(parent, 'text:date', attrib={'text:style-name': self.rststyle(style_name), 'style:data-style-name': 'rst-date-style-%d' % self.style_index})
        el2 = SubElement(automatic_styles, 'number:date-style', attrib={'style:name': 'rst-date-style-%d' % self.style_index, 'number:automatic-order': 'true', 'xmlns:number': SNSD['number'], 'xmlns:style': SNSD['style']})
        el3 = SubElement(el2, 'number:month', attrib={'number:style': 'long'})
        el3 = SubElement(el2, 'number:text')
        el3.text = '/'
        el3 = SubElement(el2, 'number:day', attrib={'number:style': 'long'})
        el3 = SubElement(el2, 'number:text')
        el3.text = '/'
        el3 = SubElement(el2, 'number:year')
    elif text == 'd2':
        self.style_index += 1
        el1 = SubElement(parent, 'text:date', attrib={'text:style-name': self.rststyle(style_name), 'style:data-style-name': 'rst-date-style-%d' % self.style_index})
        el2 = SubElement(automatic_styles, 'number:date-style', attrib={'style:name': 'rst-date-style-%d' % self.style_index, 'number:automatic-order': 'true', 'xmlns:number': SNSD['number'], 'xmlns:style': SNSD['style']})
        el3 = SubElement(el2, 'number:month', attrib={'number:style': 'long'})
        el3 = SubElement(el2, 'number:text')
        el3.text = '/'
        el3 = SubElement(el2, 'number:day', attrib={'number:style': 'long'})
        el3 = SubElement(el2, 'number:text')
        el3.text = '/'
        el3 = SubElement(el2, 'number:year', attrib={'number:style': 'long'})
    elif text == 'd3':
        self.style_index += 1
        el1 = SubElement(parent, 'text:date', attrib={'text:style-name': self.rststyle(style_name), 'style:data-style-name': 'rst-date-style-%d' % self.style_index})
        el2 = SubElement(automatic_styles, 'number:date-style', attrib={'style:name': 'rst-date-style-%d' % self.style_index, 'number:automatic-order': 'true', 'xmlns:number': SNSD['number'], 'xmlns:style': SNSD['style']})
        el3 = SubElement(el2, 'number:month', attrib={'number:textual': 'true'})
        el3 = SubElement(el2, 'number:text')
        el3.text = ' '
        el3 = SubElement(el2, 'number:day', attrib={})
        el3 = SubElement(el2, 'number:text')
        el3.text = ', '
        el3 = SubElement(el2, 'number:year', attrib={'number:style': 'long'})
    elif text == 'd4':
        self.style_index += 1
        el1 = SubElement(parent, 'text:date', attrib={'text:style-name': self.rststyle(style_name), 'style:data-style-name': 'rst-date-style-%d' % self.style_index})
        el2 = SubElement(automatic_styles, 'number:date-style', attrib={'style:name': 'rst-date-style-%d' % self.style_index, 'number:automatic-order': 'true', 'xmlns:number': SNSD['number'], 'xmlns:style': SNSD['style']})
        el3 = SubElement(el2, 'number:month', attrib={'number:textual': 'true', 'number:style': 'long'})
        el3 = SubElement(el2, 'number:text')
        el3.text = ' '
        el3 = SubElement(el2, 'number:day', attrib={})
        el3 = SubElement(el2, 'number:text')
        el3.text = ', '
        el3 = SubElement(el2, 'number:year', attrib={'number:style': 'long'})
    elif text == 'd5':
        self.style_index += 1
        el1 = SubElement(parent, 'text:date', attrib={'text:style-name': self.rststyle(style_name), 'style:data-style-name': 'rst-date-style-%d' % self.style_index})
        el2 = SubElement(automatic_styles, 'number:date-style', attrib={'style:name': 'rst-date-style-%d' % self.style_index, 'xmlns:number': SNSD['number'], 'xmlns:style': SNSD['style']})
        el3 = SubElement(el2, 'number:year', attrib={'number:style': 'long'})
        el3 = SubElement(el2, 'number:text')
        el3.text = '-'
        el3 = SubElement(el2, 'number:month', attrib={'number:style': 'long'})
        el3 = SubElement(el2, 'number:text')
        el3.text = '-'
        el3 = SubElement(el2, 'number:day', attrib={'number:style': 'long'})
    elif text == 's':
        el1 = SubElement(parent, 'text:subject', attrib={'text:style-name': self.rststyle(style_name)})
    elif text == 't':
        el1 = SubElement(parent, 'text:title', attrib={'text:style-name': self.rststyle(style_name)})
    elif text == 'a':
        el1 = SubElement(parent, 'text:author-name', attrib={'text:fixed': 'false'})
    else:
        el1 = None
    return el1