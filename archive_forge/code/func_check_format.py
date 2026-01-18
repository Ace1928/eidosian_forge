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
def check_format(self, document):
    if self.Override_default_compilation:
        return
    self.MediaBox = self.MediaBox or PDFArray(self.Rotate in (90, 270) and [0, 0, self.pageheight, self.pagewidth] or [0, 0, self.pagewidth, self.pageheight])
    if not self.Annots:
        self.Annots = None
    elif not isinstance(self.Annots, PDFObject):
        self.Annots = PDFArray(self.Annots)
    if not self.Contents:
        stream = self.stream
        if not stream:
            self.Contents = teststream()
        else:
            S = PDFStream()
            if self.compression:
                S.filters = rl_config.useA85 and [PDFBase85Encode, PDFZCompress] or [PDFZCompress]
            S.content = stream
            S.__Comment__ = 'page stream'
            self.Contents = S
    if not self.Resources:
        resources = PDFResourceDictionary()
        resources.basicFonts()
        if self.hasImages:
            resources.allProcs()
        else:
            resources.basicProcs()
        if self.XObjects:
            resources.XObject = self.XObjects
        if self.ExtGState:
            resources.ExtGState = self.ExtGState
        resources.setShading(self._shadingUsed)
        resources.setColorSpace(self._colorsUsed)
        self.Resources = resources
    if not self.Parent:
        pages = document.Pages
        self.Parent = document.Reference(pages)