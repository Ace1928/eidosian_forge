import sys, os
from binascii import hexlify, unhexlify
from hashlib import md5
from io import BytesIO
from reportlab.lib.utils import asBytes, int2Byte, rawBytes, asNative
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfdoc import PDFObject
from reportlab.platypus.flowables import Flowable
from reportlab import rl_config
def encryptPdfInMemory(inputPDF, userPassword, ownerPassword=None, canPrint=1, canModify=1, canCopy=1, canAnnotate=1, strength=40):
    """accepts a PDF file 'as a byte array in memory'; return encrypted one.

    This is a high level convenience and does not touch the hard disk in any way.
    If you are encrypting the same file over and over again, it's better to use
    pageCatcher and cache the results."""
    try:
        from rlextra.pageCatcher.pageCatcher import storeFormsInMemory, restoreFormsInMemory
    except ImportError:
        raise ImportError('reportlab.lib.pdfencrypt.encryptPdfInMemory failed because rlextra cannot be imported.\nSee https://www.reportlab.com/downloads')
    bboxInfo, pickledForms = storeFormsInMemory(inputPDF, all=1, BBoxes=1)
    names = list(bboxInfo.keys())
    firstPageSize = bboxInfo['PageForms0'][2:]
    buf = BytesIO()
    canv = Canvas(buf, pagesize=firstPageSize)
    if CLOBBERID:
        canv._doc._ID = '[(xxxxxxxxxxxxxxxx)(xxxxxxxxxxxxxxxx)]'
    formNames = restoreFormsInMemory(pickledForms, canv)
    for formName in formNames:
        canv.setPageSize(bboxInfo[formName][2:])
        canv.doForm(formName)
        canv.showPage()
    encryptCanvas(canv, userPassword, ownerPassword, canPrint, canModify, canCopy, canAnnotate, strength=strength)
    canv.save()
    return buf.getvalue()