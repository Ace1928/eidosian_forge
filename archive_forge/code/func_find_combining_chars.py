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
def find_combining_chars(text):
    """Return indices of all combining chars in  Unicode string `text`.

    >>> from docutils.utils import find_combining_chars
    >>> find_combining_chars(u'A t̆ab̆lĕ')
    [3, 6, 9]

    """
    if isinstance(text, str) and sys.version_info < (3, 0):
        return []
    return [i for i, c in enumerate(text) if unicodedata.combining(c)]