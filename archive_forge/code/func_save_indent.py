import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, LexerContext, \
from pygments.token import Text, Comment, Keyword, Name, String, Number, \
def save_indent(token_class, start=False):
    """Save a possible indentation level."""

    def callback(lexer, match, context):
        text = match.group()
        extra = ''
        if start:
            context.next_indent = len(text)
            if context.next_indent < context.indent:
                while context.next_indent < context.indent:
                    context.indent = context.indent_stack.pop()
                if context.next_indent > context.indent:
                    extra = text[context.indent:]
                    text = text[:context.indent]
        else:
            context.next_indent += len(text)
        if text:
            yield (match.start(), token_class, text)
        if extra:
            yield (match.start() + len(text), token_class.Error, extra)
        context.pos = match.end()
    return callback