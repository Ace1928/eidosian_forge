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
def generate_labeled_line(self, node, label):
    label = '%s:' % (self.language.labels[label],)
    el = self.append_p('textbody')
    el1 = SubElement(el, 'text:span', attrib={'text:style-name': self.rststyle('strong')})
    el1.text = label
    el1.tail = node.astext()
    return el