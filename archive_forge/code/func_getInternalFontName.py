import binascii, codecs, zlib
from collections import OrderedDict
from reportlab.pdfbase import pdfutils
from reportlab import rl_config
from reportlab.lib.utils import open_for_read, makeFileName, isSeq, isBytes, isUnicode, _digester, isStr, bytestr, annotateException, TimeStamp
from reportlab.lib.rl_accel import escapePDF, fp_str, asciiBase85Encode, asciiBase85Decode
from reportlab.pdfbase import pdfmetrics
from hashlib import md5
from sys import stderr
import re
def getInternalFontName(self, psfontname):
    fm = self.fontMapping
    if psfontname in fm:
        return fm[psfontname]
    else:
        try:
            fontObj = pdfmetrics.getFont(psfontname)
            if fontObj._dynamicFont:
                raise PDFError('getInternalFontName(%s) called for a dynamic font' % repr(psfontname))
            fontObj.addObjects(self)
            return fm[psfontname]
        except KeyError:
            raise PDFError('Font %s not known!' % repr(psfontname))