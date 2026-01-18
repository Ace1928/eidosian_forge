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
def showPage(self):
    """Close the current page and possibly start on a new page."""
    pageWidth = self._pagesize[0]
    pageHeight = self._pagesize[1]
    cM = self._cropMarks
    code = self._code
    if cM:
        bw = max(0, getattr(cM, 'borderWidth', 36))
        if bw:
            markLast = getattr(cM, 'markLast', 1)
            ml = min(bw, max(0, getattr(cM, 'markLength', 18)))
            mw = getattr(cM, 'markWidth', 0.5)
            mc = getattr(cM, 'markColor', black)
            mg = 2 * bw - ml
            cx0 = len(code)
            if ml and mc:
                self.saveState()
                self.setStrokeColor(mc)
                self.setLineWidth(mw)
                self.lines([(bw, 0, bw, ml), (pageWidth + bw, 0, pageWidth + bw, ml), (bw, pageHeight + mg, bw, pageHeight + 2 * bw), (pageWidth + bw, pageHeight + mg, pageWidth + bw, pageHeight + 2 * bw), (0, bw, ml, bw), (pageWidth + mg, bw, pageWidth + 2 * bw, bw), (0, pageHeight + bw, ml, pageHeight + bw), (pageWidth + mg, pageHeight + bw, pageWidth + 2 * bw, pageHeight + bw)])
                self.restoreState()
                if markLast:
                    L = code[cx0:]
                    del code[cx0:]
                    cx0 = len(code)
            bleedW = max(0, getattr(cM, 'bleedWidth', 0))
            self.saveState()
            self.translate(bw - bleedW, bw - bleedW)
            if bleedW:
                self.scale(1 + 2.0 * bleedW / pageWidth, 1 + 2.0 * bleedW / pageHeight)
            C = code[cx0:]
            del code[cx0:]
            code[0:0] = C
            self.restoreState()
            if markLast:
                code.extend(L)
            pageWidth = 2 * bw + pageWidth
            pageHeight = 2 * bw + pageHeight
    code.append(' ')
    page = pdfdoc.PDFPage()
    page.pagewidth = pageWidth
    page.pageheight = pageHeight
    page.Rotate = self._pageRotation
    page.hasImages = self._currentPageHasImages
    page.setPageTransition(self._pageTransition)
    page.setCompression(self._pageCompression)
    for box in ('crop', 'art', 'bleed', 'trim'):
        size = getattr(self, '_%sBox' % box, None)
        if size:
            setattr(page, box.capitalize() + 'Box', pdfdoc.PDFArray(size))
    if self._pageDuration is not None:
        page.Dur = self._pageDuration
    strm = self._psCommandsBeforePage + [self._preamble] + code + self._psCommandsAfterPage
    page.setStream(strm)
    self._setColorSpace(page)
    self._setExtGState(page)
    self._setXObjects(page)
    self._setShadingUsed(page)
    self._setAnnotations(page)
    self._doc.addPage(page)
    if self._onPage:
        self._onPage(self._pageNumber)
    self._startPage()