from __future__ import print_function, unicode_literals
from functools import update_wrapper
import six
import pybtex.io
from pybtex.bibtex import utils
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.bibtex.names import format_name as format_bibtex_name
from pybtex.errors import report_error
from pybtex.utils import memoize
@builtin('int.to.chr$')
def int_to_chr(i):
    n = i.pop()
    try:
        char = six.unichr(n)
    except ValueError:
        raise BibTeXError('%i passed to int.to.chr$', n)
    i.push(char)