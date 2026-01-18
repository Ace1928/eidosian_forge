import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import unirange
from pygments.lexers.css import _indentation, _starts_block
from pygments.lexers.html import HtmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.ruby import RubyLexer
def pushstate_operator_kindtest_callback(lexer, match, ctx):
    yield (match.start(), Keyword, match.group(1))
    yield (match.start(), Text, match.group(2))
    yield (match.start(), Punctuation, match.group(3))
    lexer.xquery_parse_state.append('operator')
    ctx.stack.append('kindtest')
    ctx.pos = match.end()