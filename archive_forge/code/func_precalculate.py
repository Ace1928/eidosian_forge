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
def precalculate(cmapdir):
    import os
    files = os.listdir(cmapdir)
    for file in files:
        if os.path.isfile(cmapdir + os.sep + file + '.fastmap'):
            continue
        try:
            enc = CIDEncoding(file)
        except:
            print('cannot parse %s, skipping' % enc)
            continue
        enc.fastSave(cmapdir)
        print('saved %s.fastmap' % file)