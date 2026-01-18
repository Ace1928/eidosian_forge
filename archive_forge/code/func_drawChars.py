import codecs
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Flowable
from reportlab.pdfbase import pdfmetrics, cidfonts
from reportlab.graphics.shapes import Group, String, Rect
from reportlab.graphics.widgetbase import Widget
from reportlab.lib import colors
from reportlab.lib.utils import int2Byte
def drawChars(self, charList):
    """Fills boxes in order.  None means skip a box.
        Empty boxes at end get filled with gray"""
    extraNeeded = self.rows * self.charsPerRow - len(charList)
    for i in range(extraNeeded):
        charList.append(None)
    row = 0
    col = 0
    self.canv.setFont(self.fontName, self.boxSize * 0.75)
    for ch in charList:
        if ch is None:
            self.canv.setFillGray(0.9)
            self.canv.rect((1 + col) * self.boxSize, (self.rows - row - 1) * self.boxSize, self.boxSize, self.boxSize, stroke=0, fill=1)
            self.canv.setFillGray(0.0)
        else:
            try:
                self.canv.drawCentredString((col + 1.5) * self.boxSize, (self.rows - row - 0.875) * self.boxSize, ch)
            except:
                self.canv.setFillGray(0.9)
                self.canv.rect((1 + col) * self.boxSize, (self.rows - row - 1) * self.boxSize, self.boxSize, self.boxSize, stroke=0, fill=1)
                self.canv.drawCentredString((col + 1.5) * self.boxSize, (self.rows - row - 0.875) * self.boxSize, '?')
                self.canv.setFillGray(0.0)
        col = col + 1
        if col == self.charsPerRow:
            row = row + 1
            col = 0