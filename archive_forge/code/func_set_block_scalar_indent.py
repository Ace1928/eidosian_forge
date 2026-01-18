import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, LexerContext, \
from pygments.token import Text, Comment, Keyword, Name, String, Number, \
def set_block_scalar_indent(token_class):
    """Set an explicit indentation level for a block scalar."""

    def callback(lexer, match, context):
        text = match.group()
        context.block_scalar_indent = None
        if not text:
            return
        increment = match.group(1)
        if increment:
            current_indent = max(context.indent, 0)
            increment = int(increment)
            context.block_scalar_indent = current_indent + increment
        if text:
            yield (match.start(), token_class, text)
            context.pos = match.end()
    return callback