import re
from pygments.lexer import RegexLexer, include
from pygments.util import get_bool_opt, get_list_opt
from pygments.token import Text, Comment, Operator, Keyword, Name, \
def get_dialect_from_dialect_tag(self, dialect_tag):
    left_tag_delim = '(*!'
    right_tag_delim = '*)'
    left_tag_delim_len = len(left_tag_delim)
    right_tag_delim_len = len(right_tag_delim)
    indicator_start = left_tag_delim_len
    indicator_end = -right_tag_delim_len
    if len(dialect_tag) > left_tag_delim_len + right_tag_delim_len and dialect_tag.startswith(left_tag_delim) and dialect_tag.endswith(right_tag_delim):
        indicator = dialect_tag[indicator_start:indicator_end]
        for index in range(1, len(self.dialects)):
            if indicator == self.dialects[index]:
                return indicator
        else:
            return 'unknown'
    else:
        return 'unknown'