from __future__ import print_function, unicode_literals
from functools import update_wrapper
import six
import pybtex.io
from pybtex.bibtex import utils
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.bibtex.names import format_name as format_bibtex_name
from pybtex.errors import report_error
from pybtex.utils import memoize
@builtin('<')
def operator_less(i):
    arg1 = i.pop()
    arg2 = i.pop()
    if arg2 < arg1:
        i.push(1)
    else:
        i.push(0)