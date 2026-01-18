from __future__ import print_function
from __future__ import unicode_literals
import logging
from operator import itemgetter as _itemgetter
import re
import sys
from cmakelang import lex
from cmakelang.common import UserError, InternalError
def pargs_are_full(npargs, nconsumed):
    if isinstance(npargs, int):
        return nconsumed >= npargs
    assert isinstance(npargs, STRING_TYPES), 'Unexpected npargs type {}'.format(type(npargs))
    if npargs == '?':
        return nconsumed >= 1
    if npargs in ('*', '+'):
        return False
    if npargs.endswith('+'):
        try:
            _ = int(npargs[:-1])
        except ValueError:
            raise ValueError('Unexepected npargs {}'.format(npargs))
    return False