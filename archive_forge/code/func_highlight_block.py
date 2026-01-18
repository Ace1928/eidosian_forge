from functools import partial
from importlib import import_module
from typing import Any, Dict, Optional, Type, Union
from pygments import highlight
from pygments.filters import ErrorToken
from pygments.formatter import Formatter
from pygments.formatters import HtmlFormatter, LatexFormatter
from pygments.lexer import Lexer
from pygments.lexers import (CLexer, PythonConsoleLexer, PythonLexer, RstLexer, TextLexer,
from pygments.style import Style
from pygments.styles import get_style_by_name
from pygments.util import ClassNotFound
from sphinx.locale import __
from sphinx.pygments_styles import NoneStyle, SphinxStyle
from sphinx.util import logging, texescape
def highlight_block(self, source: str, lang: str, opts: Optional[Dict]=None, force: bool=False, location: Any=None, **kwargs: Any) -> str:
    if not isinstance(source, str):
        source = source.decode()
    lexer = self.get_lexer(source, lang, opts, force, location)
    formatter = self.get_formatter(**kwargs)
    try:
        hlsource = highlight(source, lexer, formatter)
    except ErrorToken:
        if lang == 'default':
            pass
        else:
            logger.warning(__('Could not lex literal_block as "%s". Highlighting skipped.'), lang, type='misc', subtype='highlighting_failure', location=location)
        lexer = self.get_lexer(source, 'none', opts, force, location)
        hlsource = highlight(source, lexer, formatter)
    if self.dest == 'html':
        return hlsource
    else:
        return texescape.hlescape(hlsource, self.latex_engine)