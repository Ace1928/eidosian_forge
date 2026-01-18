import re
from pygments.lexer import RegexLexer, default, words, bygroups, include, using
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
def call_indent(level):
    return (_rx_indent(level), String.Doc, 'indent%s' % level)