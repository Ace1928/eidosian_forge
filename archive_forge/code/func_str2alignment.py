from reportlab.lib.colors import black
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.fonts import tt2ps
from reportlab.rl_config import canvas_basefontname as _baseFontName, \
def str2alignment(v, __map__=dict(centre=TA_CENTER, center=TA_CENTER, left=TA_LEFT, right=TA_RIGHT, justify=TA_JUSTIFY)):
    _ = __map__.get(v.lower(), None)
    if _ is not None:
        return _
    else:
        raise ValueError(f'{v!r} is illegal value for alignment')