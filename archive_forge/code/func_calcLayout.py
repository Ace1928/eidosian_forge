import codecs
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Flowable
from reportlab.pdfbase import pdfmetrics, cidfonts
from reportlab.graphics.shapes import Group, String, Rect
from reportlab.graphics.widgetbase import Widget
from reportlab.lib import colors
from reportlab.lib.utils import int2Byte
def calcLayout(self):
    """Work out x and y positions for drawing"""
    rows = self.codePoints * 1.0 / self.charsPerRow
    if rows == int(rows):
        self.rows = int(rows)
    else:
        self.rows = int(rows) + 1
    self.width = self.boxSize * (1 + self.charsPerRow)
    self.height = self.boxSize * (1 + self.rows)
    self.ylist = []
    for row in range(self.rows + 2):
        self.ylist.append(row * self.boxSize)
    self.xlist = []
    for col in range(self.charsPerRow + 2):
        self.xlist.append(col * self.boxSize)