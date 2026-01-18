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
def parseCMAPFile(self, name):
    """This is a tricky one as CMAP files are Postscript
        ones.  Some refer to others with a 'usecmap'
        command"""
    cmapfile = findCMapFile(name)
    rawdata = open(cmapfile, 'r').read()
    self._mapFileHash = self._hash(rawdata)
    usecmap_pos = rawdata.find('usecmap')
    if usecmap_pos > -1:
        chunk = rawdata[0:usecmap_pos]
        words = chunk.split()
        otherCMAPName = words[-1]
        self.parseCMAPFile(otherCMAPName)
    words = rawdata.split()
    while words != []:
        if words[0] == 'begincodespacerange':
            words = words[1:]
            while words[0] != 'endcodespacerange':
                strStart, strEnd, words = (words[0], words[1], words[2:])
                start = int(strStart[1:-1], 16)
                end = int(strEnd[1:-1], 16)
                self._codeSpaceRanges.append((start, end))
        elif words[0] == 'beginnotdefrange':
            words = words[1:]
            while words[0] != 'endnotdefrange':
                strStart, strEnd, strValue = words[0:3]
                start = int(strStart[1:-1], 16)
                end = int(strEnd[1:-1], 16)
                value = int(strValue)
                self._notDefRanges.append((start, end, value))
                words = words[3:]
        elif words[0] == 'begincidrange':
            words = words[1:]
            while words[0] != 'endcidrange':
                strStart, strEnd, strValue = words[0:3]
                start = int(strStart[1:-1], 16)
                end = int(strEnd[1:-1], 16)
                value = int(strValue)
                offset = 0
                while start + offset <= end:
                    self._cmap[start + offset] = value + offset
                    offset = offset + 1
                words = words[3:]
        else:
            words = words[1:]