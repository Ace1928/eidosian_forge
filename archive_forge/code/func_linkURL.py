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
def linkURL(self, url, rect, relative=0, thickness=0, color=None, dashArray=None, kind='URI', **kw):
    """Create a rectangular URL 'hotspot' in the given rectangle.

        if relative=1, this is in the current coord system, otherwise
        in absolute page space.
        The remaining options affect the border appearance; the border is
        drawn by Acrobat, not us.  Set thickness to zero to hide it.
        Any border drawn this way is NOT part of the page stream and
        will not show when printed to a Postscript printer or distilled;
        it is safest to draw your own."""
    from reportlab.pdfbase.pdfdoc import PDFDictionary, PDFName, PDFArray, PDFString
    ann = PDFDictionary(dict=kw)
    ann['Type'] = PDFName('Annot')
    ann['Subtype'] = PDFName('Link')
    ann['Rect'] = PDFArray(self._absRect(rect, relative))
    A = PDFDictionary()
    A['Type'] = PDFName('Action')
    uri = PDFString(url)
    A['S'] = PDFName(kind)
    if kind == 'URI':
        A['URI'] = uri
    elif kind == 'GoToR':
        A['F'] = uri
        A['D'] = '[ 0 /XYZ null null null ]'
    else:
        raise ValueError("Unknown linkURI kind '%s'" % kind)
    ann['A'] = A
    _annFormat(ann, color, thickness, dashArray)
    self._addAnnotation(ann)