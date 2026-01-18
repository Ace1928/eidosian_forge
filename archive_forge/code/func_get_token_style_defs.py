import functools
import os
import sys
import os.path
from io import StringIO
from pygments.formatter import Formatter
from pygments.token import Token, Text, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, get_list_opt
def get_token_style_defs(self, arg=None):
    prefix = self.get_css_prefix(arg)
    styles = [(level, ttype, cls, style) for cls, (style, ttype, level) in self.class2style.items() if cls and style]
    styles.sort()
    lines = ['%s { %s } /* %s */' % (prefix(cls), style, repr(ttype)[6:]) for level, ttype, cls, style in styles]
    return lines