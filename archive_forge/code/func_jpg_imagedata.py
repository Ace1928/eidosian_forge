import os
import reportlab
from reportlab import rl_config
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase import pdfdoc
from reportlab.lib.utils import isStr
from reportlab.lib.rl_accel import fp_str, asciiBase85Encode
from reportlab.lib.boxstuff import aspectRatioFix
def jpg_imagedata(self):
    fp = open(self.image, 'rb')
    try:
        result = self._jpg_imagedata(fp)
    finally:
        fp.close()
    return result