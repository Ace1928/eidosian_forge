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
def loadImageFromRaw(self, source):
    IMG = []
    imagedata = pdfutils.makeRawImage(source, IMG=IMG, detectJpeg=True)
    if not imagedata:
        return self.loadImageFromSRC(IMG[0])
    words = imagedata[1].split()
    self.width = int(words[1])
    self.height = int(words[3])
    self.colorSpace = {'/RGB': 'DeviceRGB', '/G': 'DeviceGray', '/CMYK': 'DeviceCMYK'}[words[7]]
    self.bitsPerComponent = 8
    self._filters = ('FlateDecode',)
    if IMG:
        self._checkTransparency(IMG[0])
    elif self.mask == 'auto':
        self.mask = None
    self.streamContent = b''.join(imagedata[3:-1])