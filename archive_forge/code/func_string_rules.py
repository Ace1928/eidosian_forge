from pygments.lexer import include, RegexLexer, words
from pygments.token import Comment, Keyword, Name, Number, Operator, \
def string_rules(quote_mark):
    return [('[^{}\\\\]'.format(quote_mark), String), ('\\\\.', String.Escape), (quote_mark, String, '#pop')]