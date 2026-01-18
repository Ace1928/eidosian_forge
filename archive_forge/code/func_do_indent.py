import re
from pygments.lexer import RegexLexer, default, words, bygroups, include, using
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
def do_indent(level):
    return [(_rx_indent(level), String.Doc), ('\\s*\\n', Text), default('#pop:2')]