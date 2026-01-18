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
def resetTransforms(self):
    """I want to draw something (eg, string underlines) w.r.t. the default user space.
           Reset the matrix! This should be used usually as follows::
           
              canv.saveState()
              canv.resetTransforms()
              #...draw some stuff in default space coords...
              canv.restoreState() # go back!
        """
    selfa, selfb, selfc, selfd, selfe, selff = self._currentMatrix
    det = selfa * selfd - selfc * selfb
    resulta = selfd / det
    resultc = -selfc / det
    resulte = (selfc * selff - selfd * selfe) / det
    resultd = selfa / det
    resultb = -selfb / det
    resultf = (selfe * selfb - selff * selfa) / det
    self.transform(resulta, resultb, resultc, resultd, resulte, resultf)