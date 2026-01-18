from pygments.lexer import ExtendedRegexLexer, LexerContext, \
from pygments.token import Comment, Keyword, Literal, Name, Number, Operator, \
def check_class_not_import(lexer, match, ctx):
    if match.group(0) == 'import':
        yield (match.start(), Keyword.Namespace, match.group(0))
        ctx.stack = ctx.stack[:-1] + ['fromimportfunc']
    else:
        yield (match.start(), Name.Class, match.group(0))
    ctx.pos = match.end()