import os
import marshal
import time
from hashlib import md5
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase._cidfontdata import allowedTypeFaces, allowedEncodings, CIDFontInfo, \
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfdoc
from reportlab.lib.rl_accel import escapePDF
from reportlab.rl_config import CMapSearchPath
from reportlab.lib.utils import isSeq, isBytes
def formatForPdf(self, text):
    from codecs import utf_16_be_encode
    if isBytes(text):
        text = text.decode('utf8')
    utfText = utf_16_be_encode(text)[0]
    encoded = escapePDF(utfText)
    return encoded