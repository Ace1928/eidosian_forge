import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import unirange
from pygments.lexers.css import _indentation, _starts_block
from pygments.lexers.html import HtmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.ruby import RubyLexer
def popstate_kindtest_callback(lexer, match, ctx):
    yield (match.start(), Punctuation, match.group(1))
    next_state = lexer.xquery_parse_state.pop()
    if next_state == 'occurrenceindicator':
        if re.match('[?*+]+', match.group(2)):
            yield (match.start(), Punctuation, match.group(2))
            ctx.stack.append('operator')
            ctx.pos = match.end()
        else:
            ctx.stack.append('operator')
            ctx.pos = match.end(1)
    else:
        ctx.stack.append(next_state)
        ctx.pos = match.end(1)