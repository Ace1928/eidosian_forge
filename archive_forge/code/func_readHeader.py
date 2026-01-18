from struct import pack, unpack, error as structError
from reportlab.lib.utils import bytestr, isUnicode, char2int, isStr, isBytes
from reportlab.pdfbase import pdfmetrics, pdfdoc
from reportlab import rl_config
from reportlab.lib.rl_accel import hex32, add32, calcChecksum, instanceStringWidthTTF
from collections import namedtuple
from io import BytesIO
import os, time
from reportlab.rl_config import register_reset
def readHeader(self):
    """read the sfnt header at the current position"""
    try:
        self.version = version = self.read_ulong()
    except:
        raise TTFError('"%s" is not a %s file: can\'t read version' % (self.filename, self.fileKind))
    if version == 1330926671:
        raise TTFError('%s file "%s": postscript outlines are not supported' % (self.fileKind, self.filename))
    if version not in self.ttfVersions:
        raise TTFError('Not a recognized TrueType font: version=0x%8.8X' % version)
    return version == self.ttfVersions[-1]