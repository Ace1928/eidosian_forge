from __future__ import print_function
from __future__ import unicode_literals
import logging
from operator import itemgetter as _itemgetter
import re
import sys
from cmakelang import lex
from cmakelang.common import UserError, InternalError
def get_min_npargs(npargs):
    """If the npargs specification string indicates a minimum number of arguments
     then return that value."""
    if npargs is None:
        return 0
    if isinstance(npargs, int):
        return npargs
    if npargs in ('*', '?'):
        return 0
    if npargs == '+':
        return 1
    if npargs.endswith('+'):
        try:
            return int(npargs[:-1])
        except ValueError:
            pass
    try:
        return int(npargs)
    except ValueError:
        pass
    raise ValueError('Unexpected npargs {}({})'.format(npargs, type(npargs).__name__))