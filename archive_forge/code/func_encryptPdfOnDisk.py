import sys, os
from binascii import hexlify, unhexlify
from hashlib import md5
from io import BytesIO
from reportlab.lib.utils import asBytes, int2Byte, rawBytes, asNative
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfdoc import PDFObject
from reportlab.platypus.flowables import Flowable
from reportlab import rl_config
def encryptPdfOnDisk(inputFileName, outputFileName, userPassword, ownerPassword=None, canPrint=1, canModify=1, canCopy=1, canAnnotate=1, strength=40):
    """Creates encrypted file OUTPUTFILENAME.  Returns size in bytes."""
    inputPDF = open(inputFileName, 'rb').read()
    outputPDF = encryptPdfInMemory(inputPDF, userPassword, ownerPassword, canPrint, canModify, canCopy, canAnnotate, strength=strength)
    open(outputFileName, 'wb').write(outputPDF)
    return len(outputPDF)