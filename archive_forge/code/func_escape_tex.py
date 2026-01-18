from __future__ import division
from pygments.formatter import Formatter
from pygments.lexer import Lexer
from pygments.token import Token, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, StringIO, xrange, \
def escape_tex(text, commandprefix):
    return text.replace('\\', '\x00').replace('{', '\x01').replace('}', '\x02').replace('\x00', '\\%sZbs{}' % commandprefix).replace('\x01', '\\%sZob{}' % commandprefix).replace('\x02', '\\%sZcb{}' % commandprefix).replace('^', '\\%sZca{}' % commandprefix).replace('_', '\\%sZus{}' % commandprefix).replace('&', '\\%sZam{}' % commandprefix).replace('<', '\\%sZlt{}' % commandprefix).replace('>', '\\%sZgt{}' % commandprefix).replace('#', '\\%sZsh{}' % commandprefix).replace('%', '\\%sZpc{}' % commandprefix).replace('$', '\\%sZdl{}' % commandprefix).replace('-', '\\%sZhy{}' % commandprefix).replace("'", '\\%sZsq{}' % commandprefix).replace('"', '\\%sZdq{}' % commandprefix).replace('~', '\\%sZti{}' % commandprefix)