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
def setup_paper(self, root_el):
    try:
        fin = os.popen('paperconf -s 2> /dev/null')
        content = fin.read()
        content = content.split()
        content = list(map(float, content))
        content = list(content)
        w, h = content
    except (IOError, ValueError):
        w, h = (612, 792)
    finally:
        fin.close()

    def walk(el):
        if el.tag == '{%s}page-layout-properties' % SNSD['style'] and '{%s}page-width' % SNSD['fo'] not in el.attrib:
            el.attrib['{%s}page-width' % SNSD['fo']] = '%.3fpt' % w
            el.attrib['{%s}page-height' % SNSD['fo']] = '%.3fpt' % h
            el.attrib['{%s}margin-left' % SNSD['fo']] = el.attrib['{%s}margin-right' % SNSD['fo']] = '%.3fpt' % (0.1 * w)
            el.attrib['{%s}margin-top' % SNSD['fo']] = el.attrib['{%s}margin-bottom' % SNSD['fo']] = '%.3fpt' % (0.1 * h)
        else:
            for subel in el.getchildren():
                walk(subel)
    walk(root_el)