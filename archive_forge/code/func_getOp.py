from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def getOp(self, tuple, engine):
    from reportlab.lib.sequencer import getSequencer
    globalsequencer = getSequencer()
    attr = self.attdict
    try:
        id = attr['id']
    except KeyError:
        id = None
    try:
        base = int(attr['base'])
    except:
        base = 0
    globalsequencer.reset(id, base)
    self.op = ''
    return ''