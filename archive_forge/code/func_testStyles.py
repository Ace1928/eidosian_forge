from reportlab.lib.colors import black
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.fonts import tt2ps
from reportlab.rl_config import canvas_basefontname as _baseFontName, \
def testStyles():
    pNormal = ParagraphStyle('Normal', None)
    pNormal.fontName = _baseFontName
    pNormal.fontSize = 12
    pNormal.leading = 14.4
    pNormal.listAttrs()
    print()
    pPre = ParagraphStyle('Literal', pNormal)
    pPre.fontName = 'Courier'
    pPre.listAttrs()
    return (pNormal, pPre)