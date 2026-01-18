from pygments.token import Token, STANDARD_TYPES
from pygments.util import add_metaclass
def style_for_token(cls, token):
    t = cls._styles[token]
    ansicolor = bgansicolor = None
    color = t[0]
    if color.startswith('#ansi'):
        ansicolor = color
        color = _ansimap[color]
    bgcolor = t[4]
    if bgcolor.startswith('#ansi'):
        bgansicolor = bgcolor
        bgcolor = _ansimap[bgcolor]
    return {'color': color or None, 'bold': bool(t[1]), 'italic': bool(t[2]), 'underline': bool(t[3]), 'bgcolor': bgcolor or None, 'border': t[5] or None, 'roman': bool(t[6]) or None, 'sans': bool(t[7]) or None, 'mono': bool(t[8]) or None, 'ansicolor': ansicolor, 'bgansicolor': bgansicolor}