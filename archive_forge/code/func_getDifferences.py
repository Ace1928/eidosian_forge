import os, sys, encodings
from reportlab.pdfbase import _fontdata
from reportlab.lib.logger import warnOnce
from reportlab.lib.utils import rl_isfile, rl_glob, rl_isdir, open_and_read, open_and_readlines, findInPaths, isSeq, isStr
from reportlab.rl_config import defaultEncoding, T1SearchPath
from reportlab.lib.rl_accel import unicode2T1, instanceStringWidthT1
from reportlab.pdfbase import rl_codecs
from reportlab.rl_config import register_reset
def getDifferences(self, otherEnc):
    """
        Return a compact list of the code points differing between two encodings

        This is in the Adobe format: list of
           [[b1, name1, name2, name3],
           [b2, name4]]
           
        where b1...bn is the starting code point, and the glyph names following
        are assigned consecutive code points.
        
        """
    ranges = []
    curRange = None
    for i in range(len(self.vector)):
        glyph = self.vector[i]
        if glyph == otherEnc.vector[i]:
            if curRange:
                ranges.append(curRange)
                curRange = []
        elif curRange:
            curRange.append(glyph)
        elif glyph:
            curRange = [i, glyph]
    if curRange:
        ranges.append(curRange)
    return ranges