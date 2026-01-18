import re
from pygments.lexer import RegexLexer, include, bygroups, default, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
def stringy(whatkind):
    return [('[^"\\\\]', whatkind), ('\\\\[\\\\"abtnvfr]', String.Escape), ('\\\\\\^[\\x40-\\x5e]', String.Escape), ('\\\\[0-9]{3}', String.Escape), ('\\\\u[0-9a-fA-F]{4}', String.Escape), ('\\\\\\s+\\\\', String.Interpol), ('"', whatkind, '#pop')]