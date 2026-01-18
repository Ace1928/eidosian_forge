import sys, os
from binascii import hexlify, unhexlify
from hashlib import md5
from io import BytesIO
from reportlab.lib.utils import asBytes, int2Byte, rawBytes, asNative
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfdoc import PDFObject
from reportlab.platypus.flowables import Flowable
from reportlab import rl_config
def permissionBits(self):
    p = 0
    if self.canPrint:
        p = p | printable
    if self.canModify:
        p = p | modifiable
    if self.canCopy:
        p = p | copypastable
    if self.canAnnotate:
        p = p | annotatable
    p = p | higherbits
    return p