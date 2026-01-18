from reportlab.lib.units import cm
from reportlab.lib.utils import commasplit, escapeOnce, encode_label, decode_label, strTypes, asUnicode, asNative
from reportlab.lib.styles import ParagraphStyle, _baseFontName
from reportlab.lib import sequencer as rl_sequencer
from reportlab.platypus.paragraph import Paragraph
from reportlab.platypus.doctemplate import IndexingFlowable
from reportlab.platypus.tables import TableStyle, Table
from reportlab.platypus.flowables import Spacer
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
import unicodedata
from ast import literal_eval
def drawTOCEntryEnd(canvas, kind, label):
    """Callback to draw dots and page numbers after each entry."""
    label = label.split(',')
    page, level, key = (int(label[0]), int(label[1]), literal_eval(label[2]))
    style = self.getLevelStyle(level)
    if self.dotsMinLevel >= 0 and level >= self.dotsMinLevel:
        dot = ' . '
    else:
        dot = ''
    if self.formatter:
        page = self.formatter(page)
    drawPageNumbers(canvas, style, [(page, key)], availWidth, availHeight, dot)