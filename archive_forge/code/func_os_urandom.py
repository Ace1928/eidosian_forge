import sys, os
from binascii import hexlify, unhexlify
from hashlib import md5
from io import BytesIO
from reportlab.lib.utils import asBytes, int2Byte, rawBytes, asNative
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfdoc import PDFObject
from reportlab.platypus.flowables import Flowable
from reportlab import rl_config
def os_urandom(n):
    global _os_random_x
    b = [_os_random_b[(i + _os_random_x) % 256] for i in range(n)]
    b = bytes(b)
    _os_random_x = (_os_random_x + n) % 256
    return b