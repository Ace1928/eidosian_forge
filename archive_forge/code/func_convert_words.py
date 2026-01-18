from __future__ import absolute_import, unicode_literals
import re
import six
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.py3compat import fix_unicode_literals_in_doctest
from pybtex.utils import pairwise
from pybtex import py3compat
def convert_words(words):
    for word in words:
        if word.startswith('\\'):
            yield word
        else:
            yield convert(word, state)