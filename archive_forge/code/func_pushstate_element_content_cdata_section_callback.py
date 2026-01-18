import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import unirange
from pygments.lexers.css import _indentation, _starts_block
from pygments.lexers.html import HtmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.ruby import RubyLexer
def pushstate_element_content_cdata_section_callback(lexer, match, ctx):
    yield (match.start(), String.Doc, match.group(1))
    ctx.stack.append('cdata_section')
    lexer.xquery_parse_state.append('element_content')
    ctx.pos = match.end()