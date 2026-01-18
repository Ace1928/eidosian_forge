from __future__ import absolute_import, unicode_literals
import re
import six
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.py3compat import fix_unicode_literals_in_doctest
from pybtex.utils import pairwise
from pybtex import py3compat
def find_closing_brace(self, chars):
    for char in chars:
        if char == '{':
            yield BibTeXString(chars, self.level + 1)
        elif char == '}' and self.level > 0:
            self.is_closed = True
            return
        else:
            yield char