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
def setPageDuration(self, duration=None):
    """Allows hands-off animation of presentations :-)

        If this is set to a number, in full screen mode, Acrobat Reader
        will advance to the next page after this many seconds. The
        duration of the transition itself (fade/flicker etc.) is controlled
        by the 'duration' argument to setPageTransition; this controls
        the time spent looking at the page.  This is effective for all
        subsequent pages."""
    self._pageDuration = duration