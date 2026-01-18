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
def linkRect(self, contents, destinationname, Rect=None, addtopage=1, name=None, relative=1, thickness=0, color=None, dashArray=None, **kw):
    """rectangular link annotation w.r.t the current user transform.
           if the transform is skewed/rotated the absolute rectangle will use the max/min x/y
        """
    destination = self._bookmarkReference(destinationname)
    Rect = self._absRect(Rect, relative)
    kw['Rect'] = Rect
    kw['Contents'] = contents
    kw['Destination'] = destination
    _annFormat(kw, color, thickness, dashArray)
    return self._addAnnotation(pdfdoc.LinkAnnotation(**kw), name, addtopage)