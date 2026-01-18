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
def set_align_from_classes(self, node):
    """Convert ``align-*`` class arguments into alignment args."""
    align = [cls for cls in node['classes'] if cls.startswith('align-')]
    if align:
        node['align'] = align[-1].replace('align-', '')
        node['classes'] = [cls for cls in node['classes'] if not cls.startswith('align-')]