from pygments import highlight
from pygments.formatters.html import HtmlFormatter
from pygments.lexer import bygroups
from pygments.lexer import DelegatingLexer
from pygments.lexer import include
from pygments.lexer import RegexLexer
from pygments.lexer import using
from pygments.lexers.agile import Python3Lexer
from pygments.lexers.agile import PythonLexer
from pygments.lexers.web import CssLexer
from pygments.lexers.web import HtmlLexer
from pygments.lexers.web import JavascriptLexer
from pygments.lexers.web import XmlLexer
from pygments.token import Comment
from pygments.token import Keyword
from pygments.token import Name
from pygments.token import Operator
from pygments.token import Other
from pygments.token import String
from pygments.token import Text
def syntax_highlight(filename='', language=None):
    mako_lexer = MakoLexer()
    python_lexer = Python3Lexer()
    if filename.startswith('memory:') or language == 'mako':
        return lambda string: highlight(string, mako_lexer, pygments_html_formatter)
    return lambda string: highlight(string, python_lexer, pygments_html_formatter)