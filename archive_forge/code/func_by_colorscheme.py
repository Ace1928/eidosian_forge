import numba.core.config
from pygments.styles.manni import ManniStyle
from pygments.styles.monokai import MonokaiStyle
from pygments.styles.native import NativeStyle
from pygments.lexer import RegexLexer, include, bygroups, words
from pygments.token import Text, Name, String,  Punctuation, Keyword, \
from pygments.style import Style
def by_colorscheme():
    """
    Get appropriate style for highlighting according to
    NUMBA_COLOR_SCHEME setting
    """
    styles = DefaultStyle.styles.copy()
    styles.update({Name.Variable: '#888888'})
    custom_default = type('CustomDefaultStyle', (Style,), {'styles': styles})
    style_map = {'no_color': custom_default, 'dark_bg': MonokaiStyle, 'light_bg': ManniStyle, 'blue_bg': NativeStyle, 'jupyter_nb': DefaultStyle}
    return style_map[numba.core.config.COLOR_SCHEME]