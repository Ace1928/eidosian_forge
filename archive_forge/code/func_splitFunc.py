import os
from copy import deepcopy, copy
from reportlab.lib.colors import gray, lightgrey
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.styles import _baseFontName
from reportlab.lib.utils import strTypes, rl_safe_exec, annotateException
from reportlab.lib.abag import ABag
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.rl_config import _FUZZ, overlapAttachedSpace, ignoreContainerActions, listWrapOnFakeWidth
from reportlab.lib.sequencer import _type2formatter
from reportlab.lib.styles import ListStyle
def splitFunc(ah, endSlack=0):
    if ah not in _fres:
        c = []
        w = 0
        h = 0
        cn = None
        icheck = nCols - 2 if endSlack else -1
        for i in range(nCols):
            wi, hi, c0, c1 = self._findSplit(canv, cw, ah, content=cn, paraFix=False)
            w = max(w, wi)
            h = max(h, hi)
            c.append(c0)
            if i == icheck:
                wc, hc, cc0, cc1 = self._findSplit(canv, cw, 2 * ah, content=c1, paraFix=False)
                if hc <= (1 + endSlack) * ah:
                    c.append(c1)
                    h = ah - 1e-06
                    cn = []
                    break
            cn = c1
        _fres[ah] = (ah + 100000 * int(cn != []), cn == [], (w, h, c, cn))
    return _fres[ah][2]