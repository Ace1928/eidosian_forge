import codecs
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Flowable
from reportlab.pdfbase import pdfmetrics, cidfonts
from reportlab.graphics.shapes import Group, String, Rect
from reportlab.graphics.widgetbase import Widget
from reportlab.lib import colors
from reportlab.lib.utils import int2Byte
def makeRow(self, row):
    """Works out the character values for this Big5 row.
        Rows start at 0xA1"""
    cells = []
    if self.encodingName.find('B5') > -1:
        for y in [4, 5, 6, 7, 10, 11, 12, 13, 14, 15]:
            for x in range(16):
                col = y * 16 + x
                ch = int2Byte(row) + int2Byte(col)
                cells.append(ch)
    else:
        cells.append([None] * 160)
    return cells