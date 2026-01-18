import sys
import os.path
import re
import urllib.request, urllib.parse, urllib.error
import docutils
from docutils import nodes, utils, writers, languages, io
from docutils.utils.error_reporting import SafeString
from docutils.transforms import writer_aux
from docutils.utils.math import (unichar2tex, pick_math_environment,
def apply_template(self):
    template_file = open(self.document.settings.template, 'rb')
    template = str(template_file.read(), 'utf-8')
    template_file.close()
    subs = self.interpolation_dict()
    return template % subs