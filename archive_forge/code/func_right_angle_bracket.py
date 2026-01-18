from pygments.lexer import ExtendedRegexLexer, bygroups, DelegatingLexer
from pygments.token import Name, Number, String, Comment, Punctuation, \
def right_angle_bracket(lexer, match, ctx):
    if len(ctx.stack) > 1 and ctx.stack[-2] == 'string':
        ctx.stack.pop()
    yield (match.start(), String.Interpol, '}')
    ctx.pos = match.end()
    pass