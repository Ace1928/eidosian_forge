from __future__ import division
from pygments.formatter import Formatter
from pygments.lexer import Lexer
from pygments.token import Token, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, StringIO, xrange, \
def rgbcolor(col):
    if col:
        return ','.join(['%.2f' % (int(col[i] + col[i + 1], 16) / 255.0) for i in (0, 2, 4)])
    else:
        return '1,1,1'