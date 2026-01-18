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
def loadImageFromSRC(self, im):
    """Extracts the stream, width and height"""
    fp = im.jpeg_fh()
    if fp:
        self.loadImageFromJPEG(fp)
    else:
        self.width, self.height = im.getSize()
        raw = im.getRGBData()
        self.streamContent = zlib.compress(raw)
        if rl_config.useA85:
            self.streamContent = asciiBase85Encode(self.streamContent)
            self._filters = ('ASCII85Decode', 'FlateDecode')
        else:
            self._filters = ('FlateDecode',)
        self.colorSpace = _mode2CS[im.mode]
        self.bitsPerComponent = 8
        self._checkTransparency(im)