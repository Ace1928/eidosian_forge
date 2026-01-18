import sys
import os.path
import re
import urllib.request, urllib.parse, urllib.error
import docutils
from docutils import nodes, utils, writers, languages, io
from docutils.utils.error_reporting import SafeString
from docutils.transforms import writer_aux
from docutils.utils.math import (unichar2tex, pick_math_environment,
def add_meta(self, tag):
    self.meta.append(tag)
    self.head.append(tag)