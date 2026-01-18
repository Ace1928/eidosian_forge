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
def getData(self):
    """Simple persistence helper.  Return a dict with all that matters."""
    return {'mapFileHash': self._mapFileHash, 'codeSpaceRanges': self._codeSpaceRanges, 'notDefRanges': self._notDefRanges, 'cmap': self._cmap}