import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import unirange
from pygments.lexers.css import _indentation, _starts_block
from pygments.lexers.html import HtmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.ruby import RubyLexer
def popstate_callback(lexer, match, ctx):
    yield (match.start(), Punctuation, match.group(1))
    if len(lexer.xquery_parse_state) == 0:
        ctx.stack.pop()
    elif len(ctx.stack) > 1:
        ctx.stack.append(lexer.xquery_parse_state.pop())
    else:
        ctx.stack = ['root']
    ctx.pos = match.end()