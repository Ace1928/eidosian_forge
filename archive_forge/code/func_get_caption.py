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
def get_caption(self):
    if not self.caption:
        return ''
    caption = ''.join(self.caption)
    if 1 == self._translator.thead_depth():
        return '\\caption{%s}\\\\\n' % caption
    return '\\caption[]{%s (... continued)}\\\\\n' % caption