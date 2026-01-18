from reportlab.platypus.flowables import Flowable, Preformatted
from reportlab import rl_config
from reportlab.lib.styles import PropertySet, ParagraphStyle, _baseFontName
from reportlab.lib import colors
from reportlab.lib.utils import annotateException, IdentStr, flatten, isStr, asNative, strTypes, __UNSET__
from reportlab.lib.validators import isListOfNumbersOrNone
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.abag import ABag as CellFrame
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus.doctemplate import Indenter, NullActionFlowable
from reportlab.platypus.flowables import LIIndenter
from collections import namedtuple
def spanFixDim(V0, V, spanCons, lim=None, FUZZ=rl_config._FUZZ):
    M = {}
    if not lim:
        lim = len(V0)
    for v, (x0, x1) in reversed(sorted(((iv, ik) for ik, iv in spanCons.items()))):
        if x0 >= lim:
            continue
        x1 += 1
        t = sum([V[x] + M.get(x, 0) for x in range(x0, x1)])
        if t >= v - FUZZ:
            continue
        X = [x for x in range(x0, x1) if V0[x] is None]
        if not X:
            continue
        v -= t
        v /= float(len(X))
        for x in X:
            M[x] = M.get(x, 0) + v
    for x, v in M.items():
        V[x] += v