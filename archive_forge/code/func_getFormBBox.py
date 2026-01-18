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
def getFormBBox(self, name, boxType='MediaBox'):
    """get the declared bounding box of the form as a list.
        If you specify a different PDF box definition (e.g. the
        ArtBox) and it has one, that's what you'll get."""
    internalname = xObjectName(name)
    if internalname in self.idToObject:
        theform = self.idToObject[internalname]
        if hasattr(theform, '_extra_pageCatcher_info'):
            return theform._extra_pageCatcher_info[boxType]
        if isinstance(theform, PDFFormXObject):
            return theform.BBoxList()
        elif isinstance(theform, PDFStream):
            return list(theform.dictionary.dict[boxType].sequence)
        else:
            raise ValueError("I don't understand the form instance %s" % repr(name))