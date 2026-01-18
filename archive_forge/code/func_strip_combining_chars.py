import sys
import os
import os.path
import re
import itertools
import warnings
import unicodedata
from docutils import ApplicationError, DataError, __version_info__
from docutils import nodes
from docutils.nodes import unescape
import docutils.io
from docutils.utils.error_reporting import ErrorOutput, SafeString
def strip_combining_chars(text):
    if isinstance(text, str) and sys.version_info < (3, 0):
        return text
    return ''.join([c for c in text if not unicodedata.combining(c)])