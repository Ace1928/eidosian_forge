import functools
import os
import sys
import os.path
from io import StringIO
from pygments.formatter import Formatter
from pygments.token import Token, Text, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, get_list_opt
def get_linenos_style_defs(self):
    lines = ['pre { %s }' % self._pre_style, 'td.linenos .normal { %s }' % self._linenos_style, 'span.linenos { %s }' % self._linenos_style, 'td.linenos .special { %s }' % self._linenos_special_style, 'span.linenos.special { %s }' % self._linenos_special_style]
    return lines