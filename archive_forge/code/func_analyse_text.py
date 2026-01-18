import re
from pygments.lexer import RegexLexer, default, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
def analyse_text(text):
    if text.startswith('(*'):
        return True