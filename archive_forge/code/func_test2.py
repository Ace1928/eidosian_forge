from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def test2(canv, testpara):
    from reportlab.lib.units import inch
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import rparsexml
    parsedpara = rparsexml.parsexmlSimple(testpara, entityReplacer=None)
    S = ParagraphStyle('Normal', None)
    P = Para(S, parsedpara)
    w, h = P.wrap(5 * inch, 10 * inch)
    print('wrapped as', (h, w))
    canv.saveState()
    canv.translate(1 * inch, 1 * inch)
    canv.rect(0, 0, 5 * inch, 10 * inch, fill=0, stroke=1)
    P.canv = canv
    canv.saveState()
    P.draw()
    canv.restoreState()
    canv.setStrokeColorRGB(1, 0, 0)
    canv.rect(0, 0, w, h, fill=0, stroke=1)
    canv.restoreState()
    canv.showPage()