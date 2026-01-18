from pygments.lexer import ExtendedRegexLexer, LexerContext, \
from pygments.token import Comment, Keyword, Literal, Name, Number, Operator, \
def store_indent(lexer, match, ctx):
    ctx.indent, _ = CleanLexer.indent_len(match.group(0))
    ctx.pos = match.end()
    yield (match.start(), Text, match.group(0))