import sys, os
from binascii import hexlify, unhexlify
from hashlib import md5
from io import BytesIO
from reportlab.lib.utils import asBytes, int2Byte, rawBytes, asNative
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfdoc import PDFObject
from reportlab.platypus.flowables import Flowable
from reportlab import rl_config
def encryptCanvas(canvas, userPassword, ownerPassword=None, canPrint=1, canModify=1, canCopy=1, canAnnotate=1, strength=40):
    """Applies encryption to the document being generated"""
    enc = StandardEncryption(userPassword, ownerPassword, canPrint, canModify, canCopy, canAnnotate, strength=strength)
    canvas._doc.encrypt = enc