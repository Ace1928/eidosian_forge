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
def loadImageFromA85(self, source):
    IMG = []
    imagedata = pdfutils.makeA85Image(source, IMG=IMG, detectJpeg=True)
    if not imagedata:
        return self.loadImageFromSRC(IMG[0])
    imagedata = [s.strip() for s in imagedata]
    words = imagedata[1].split()
    self.width, self.height = (int(words[1]), int(words[3]))
    self.colorSpace = {'/RGB': 'DeviceRGB', '/G': 'DeviceGray', '/CMYK': 'DeviceCMYK'}[words[7]]
    self.bitsPerComponent = 8
    self._filters = ('ASCII85Decode', 'FlateDecode')
    if IMG:
        self._checkTransparency(IMG[0])
    elif self.mask == 'auto':
        self.mask = None
    self.streamContent = ''.join(imagedata[3:-1])