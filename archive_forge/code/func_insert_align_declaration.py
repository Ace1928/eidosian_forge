import sys
import os
import time
import re
import string
import urllib.request, urllib.parse, urllib.error
from docutils import frontend, nodes, languages, writers, utils, io
from docutils.utils.error_reporting import SafeString
from docutils.transforms import writer_aux
from docutils.utils.math import pick_math_environment, unichar2tex
def insert_align_declaration(self, node, default=None):
    align = node.get('align', default)
    if align == 'left':
        self.out.append('\\raggedright\n')
    elif align == 'center':
        self.out.append('\\centering\n')
    elif align == 'right':
        self.out.append('\\raggedleft\n')