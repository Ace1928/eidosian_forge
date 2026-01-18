import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, LexerContext, \
from pygments.token import Text, Comment, Keyword, Name, String, Number, \
def parse_plain_scalar_indent(token_class):
    """Process indentation spaces in a plain scalar."""

    def callback(lexer, match, context):
        text = match.group()
        if len(text) <= context.indent:
            context.stack.pop()
            context.stack.pop()
            return
        if text:
            yield (match.start(), token_class, text)
            context.pos = match.end()
    return callback