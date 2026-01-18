import math
import xmllib
from rdkit.sping.pid import Font
from sping.PDF import PDFCanvas
def calcNewFont(self, font):
    """Given a font (does not accept font==None), creates a                 new font based on the format of this text segment."""
    if self.greek:
        face = 'symbol'
    else:
        face = font.face
    return Font(face=face, size=font.size - self.super * sizedelta - self.sub * sizedelta, underline=self.underline or font.underline, bold=self.bold or font.bold, italic=self.italic or font.italic)