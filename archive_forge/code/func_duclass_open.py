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
def duclass_open(self, node):
    """Open a group and insert declarations for class values."""
    if not isinstance(node.parent, nodes.compound):
        self.out.append('\n')
    for cls in node['classes']:
        if cls.startswith('language-'):
            language = self.babel.language_name(cls[9:])
            if language:
                self.babel.otherlanguages[language] = True
                self.out.append('\\begin{selectlanguage}{%s}\n' % language)
        else:
            self.fallbacks['DUclass'] = PreambleCmds.duclass
            self.out.append('\\begin{DUclass}{%s}\n' % cls)