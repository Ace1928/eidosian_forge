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
def language_name(self, language_code):
    """Return TeX language name for `language_code`"""
    for tag in utils.normalize_language_tag(language_code):
        try:
            return self.language_codes[tag]
        except KeyError:
            pass
    if self.reporter is not None:
        self.reporter.warning(self.warn_msg % language_code)
    return ''