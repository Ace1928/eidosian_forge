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
def get_opening(self, width='\\linewidth'):
    align_map = {'left': 'l', 'center': 'c', 'right': 'r'}
    align = align_map.get(self.get('align') or 'center')
    opening = ['\\begin{%s}[%s]' % (self.get_latex_type(), align)]
    if not self.colwidths_auto:
        opening.insert(0, '\\setlength{\\DUtablewidth}{%s}' % width)
    return '\n'.join(opening)