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
def create_meta(self):
    if WhichElementTree == 'lxml':
        root = Element('office:document-meta', nsmap=META_NAMESPACE_DICT, nsdict=META_NAMESPACE_DICT)
    else:
        root = Element('office:document-meta', attrib=META_NAMESPACE_ATTRIB, nsdict=META_NAMESPACE_DICT)
    doc = etree.ElementTree(root)
    root = SubElement(root, 'office:meta', nsdict=METNSD)
    el1 = SubElement(root, 'meta:generator', nsdict=METNSD)
    el1.text = 'Docutils/rst2odf.py/%s' % (VERSION,)
    s1 = os.environ.get('USER', '')
    el1 = SubElement(root, 'meta:initial-creator', nsdict=METNSD)
    el1.text = s1
    s2 = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())
    el1 = SubElement(root, 'meta:creation-date', nsdict=METNSD)
    el1.text = s2
    el1 = SubElement(root, 'dc:creator', nsdict=METNSD)
    el1.text = s1
    el1 = SubElement(root, 'dc:date', nsdict=METNSD)
    el1.text = s2
    el1 = SubElement(root, 'dc:language', nsdict=METNSD)
    el1.text = 'en-US'
    el1 = SubElement(root, 'meta:editing-cycles', nsdict=METNSD)
    el1.text = '1'
    el1 = SubElement(root, 'meta:editing-duration', nsdict=METNSD)
    el1.text = 'PT00M01S'
    title = self.visitor.get_title()
    el1 = SubElement(root, 'dc:title', nsdict=METNSD)
    if title:
        el1.text = title
    else:
        el1.text = '[no title]'
    meta_dict = self.visitor.get_meta_dict()
    keywordstr = meta_dict.get('keywords')
    if keywordstr is not None:
        keywords = split_words(keywordstr)
        for keyword in keywords:
            el1 = SubElement(root, 'meta:keyword', nsdict=METNSD)
            el1.text = keyword
    description = meta_dict.get('description')
    if description is not None:
        el1 = SubElement(root, 'dc:description', nsdict=METNSD)
        el1.text = description
    s1 = ToString(doc)
    return s1