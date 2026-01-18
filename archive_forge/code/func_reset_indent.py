import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, LexerContext, \
from pygments.token import Text, Comment, Keyword, Name, String, Number, \
def reset_indent(token_class):
    """Reset the indentation levels."""

    def callback(lexer, match, context):
        text = match.group()
        context.indent_stack = []
        context.indent = -1
        context.next_indent = 0
        context.block_scalar_indent = None
        yield (match.start(), token_class, text)
        context.pos = match.end()
    return callback