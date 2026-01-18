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
def process_footnotes(self):
    for node, el1 in self.footnote_list:
        backrefs = node.attributes.get('backrefs', [])
        first = True
        for ref in backrefs:
            el2 = self.footnote_ref_dict.get(ref)
            if el2 is not None:
                if first:
                    first = False
                    el3 = copy.deepcopy(el1)
                    el2.append(el3)
                else:
                    children = el2.getchildren()
                    if len(children) > 0:
                        child = children[0]
                        ref1 = child.text
                        attribkey = add_ns('text:id', nsdict=SNSD)
                        id1 = el2.get(attribkey, 'footnote-error')
                        if id1 is None:
                            id1 = ''
                        tag = add_ns('text:note-ref', nsdict=SNSD)
                        el2.tag = tag
                        if self.settings.endnotes_end_doc:
                            note_class = 'endnote'
                        else:
                            note_class = 'footnote'
                        el2.attrib.clear()
                        attribkey = add_ns('text:note-class', nsdict=SNSD)
                        el2.attrib[attribkey] = note_class
                        attribkey = add_ns('text:ref-name', nsdict=SNSD)
                        el2.attrib[attribkey] = id1
                        attribkey = add_ns('text:reference-format', nsdict=SNSD)
                        el2.attrib[attribkey] = 'page'
                        el2.text = ref1