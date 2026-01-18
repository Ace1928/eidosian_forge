import re
import hashlib
from string import digits
from math import sin, cos, tan, pi
from reportlab import rl_config
from reportlab.pdfbase import pdfdoc
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen  import pathobject
from reportlab.pdfgen.textobject import PDFTextObject, _PDFColorSetter
from reportlab.lib.colors import black, _chooseEnforceColorSpace, Color, CMYKColor, toColor
from reportlab.lib.utils import ImageReader, isSeq, isStr, isUnicode, _digester, asUnicode
from reportlab.lib.rl_accel import fp_str, escapePDF
from reportlab.lib.boxstuff import aspectRatioFix
def linkAbsolute(self, contents, destinationname, Rect=None, addtopage=1, name=None, thickness=0, color=None, dashArray=None, **kw):
    """rectangular link annotation positioned wrt the default user space.
           The identified rectangle on the page becomes a "hot link" which
           when clicked will send the viewer to the page and position identified
           by the destination.

           Rect identifies (lowerx, lowery, upperx, uppery) for lower left
           and upperright points of the rectangle.  Translations and other transforms
           are IGNORED (the rectangular position is given with respect
           to the default user space.
           destinationname should be the name of a bookmark (which may be defined later
           but must be defined before the document is generated).

           You may want to use the keyword argument Border='[0 0 0]' to
           suppress the visible rectangle around the during viewing link."""
    return self.linkRect(contents, destinationname, Rect, addtopage, name, relative=0, thickness=thickness, color=color, dashArray=dashArray, **kw)