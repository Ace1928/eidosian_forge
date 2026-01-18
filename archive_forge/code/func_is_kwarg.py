from __future__ import print_function
from __future__ import unicode_literals
import logging
from operator import itemgetter as _itemgetter
import re
import sys
from cmakelang import lex
from cmakelang.common import UserError, InternalError
def is_kwarg(self, key):
    subspec = self.kwargs.get(key, None)
    if subspec is None:
        return False
    if isinstance(subspec, int):
        return subspec != 0
    if isinstance(subspec, STRING_TYPES + (CommandSpec,)):
        return True
    raise ValueError('Unexpected kwargspec for {}: {}'.format(key, type(subspec)))