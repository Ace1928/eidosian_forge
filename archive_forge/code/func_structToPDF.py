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
def structToPDF(structure):
    """Converts deeply nested structure to PDFdoc dictionary/array objects"""
    if isinstance(structure, dict):
        newDict = {}
        for k, v in structure.items():
            newDict[k] = structToPDF(v)
        return pdfdoc.PDFDictionary(newDict)
    elif isSeq(structure):
        newList = []
        for elem in structure:
            newList.append(structToPDF(elem))
        return pdfdoc.PDFArray(newList)
    else:
        return structure