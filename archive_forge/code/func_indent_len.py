from pygments.lexer import ExtendedRegexLexer, LexerContext, \
from pygments.token import Comment, Keyword, Literal, Name, Number, Operator, \
@staticmethod
def indent_len(text):
    text = text.replace('\n', '')
    return (len(text.replace('\t', '    ')), len(text))