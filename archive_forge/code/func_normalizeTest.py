import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
def normalizeTest(self, x):
    try:
        self.normalize(x)
        return True
    except:
        return False