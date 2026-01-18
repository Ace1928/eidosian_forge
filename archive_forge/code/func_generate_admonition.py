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
def generate_admonition(self, node, label, title=None):
    if hasattr(self.language, 'labels'):
        translated_label = self.language.labels.get(label, label)
    else:
        translated_label = label
    el1 = SubElement(self.current_element, 'text:p', attrib={'text:style-name': self.rststyle('admon-%s-hdr', (label,))})
    if title:
        el1.text = title
    else:
        el1.text = '%s!' % (translated_label.capitalize(),)
    s1 = self.rststyle('admon-%s-body', (label,))
    self.paragraph_style_stack.append(s1)