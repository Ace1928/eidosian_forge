import os
import reportlab
from reportlab import rl_config
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase import pdfdoc
from reportlab.lib.utils import isStr
from reportlab.lib.rl_accel import fp_str, asciiBase85Encode
from reportlab.lib.boxstuff import aspectRatioFix
def drawInlineImage(self, canvas, preserveAspectRatio=False, anchor='sw', anchorAtXY=False, showBoundary=False, extraReturn=None):
    """Draw an Image into the specified rectangle.  If width and
        height are omitted, they are calculated from the image size.
        Also allow file names as well as images.  This allows a
        caching mechanism"""
    width = self.width
    height = self.height
    if width < 1e-06 or height < 1e-06:
        return False
    x, y, self.width, self.height, scaled = aspectRatioFix(preserveAspectRatio, anchor, self.x, self.y, width, height, self.imgwidth, self.imgheight, anchorAtXY)
    if not canvas.bottomup:
        y = y + height
    canvas._code.append('q %s 0 0 %s cm' % (fp_str(self.width), fp_str(self.height, x, y)))
    width = self.width
    height = self.height
    for line in self.imageData:
        canvas._code.append(line)
    canvas._code.append('Q')
    if showBoundary:
        canvas.drawBoundary(showBoundary, x, y, width, height)
    if extraReturn:
        for k in extraReturn.keys():
            extraReturn[k] = vars()[k]
    return True