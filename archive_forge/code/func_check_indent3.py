from pygments.lexer import ExtendedRegexLexer, LexerContext, \
from pygments.token import Comment, Keyword, Literal, Name, Number, Operator, \
def check_indent3(lexer, match, ctx):
    indent, reallen = CleanLexer.indent_len(match.group(0))
    if indent > ctx.indent:
        yield (match.start(), Whitespace, match.group(0))
        ctx.pos = match.start() + reallen + 1
    else:
        ctx.indent = 0
        ctx.pos = match.start()
        ctx.stack = ctx.stack[:-3]
        yield (match.start(), Whitespace, match.group(0)[1:])
        if match.group(0) == '\n\n':
            ctx.pos = ctx.pos + 1