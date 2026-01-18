import os, sys, encodings
from reportlab.pdfbase import _fontdata
from reportlab.lib.logger import warnOnce
from reportlab.lib.utils import rl_isfile, rl_glob, rl_isdir, open_and_read, open_and_readlines, findInPaths, isSeq, isStr
from reportlab.rl_config import defaultEncoding, T1SearchPath
from reportlab.lib.rl_accel import unicode2T1, instanceStringWidthT1
from reportlab.pdfbase import rl_codecs
from reportlab.rl_config import register_reset
def bruteForceSearchForAFM(faceName):
    """Looks in all AFM files on path for face with given name.

    Returns AFM file name or None.  Ouch!"""
    from reportlab.rl_config import T1SearchPath
    for dirname in T1SearchPath:
        if not rl_isdir(dirname):
            continue
        possibles = rl_glob(dirname + os.sep + '*.[aA][fF][mM]')
        for possible in possibles:
            try:
                topDict, glyphDict = parseAFMFile(possible)
                if topDict['FontName'] == faceName:
                    return possible
            except:
                t, v, b = sys.exc_info()
                v.args = (' '.join(map(str, v.args)) + ', while looking for faceName=%r' % faceName,)
                raise