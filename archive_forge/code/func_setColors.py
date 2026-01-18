import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def setColors(**kw):
    UNDEF = []
    progress = 1
    assigned = {}
    while kw and progress:
        progress = 0
        for k, v in kw.items():
            if isinstance(v, (tuple, list)):
                c = list(map(lambda x, UNDEF=UNDEF: toColor(x, UNDEF), v))
                if isinstance(v, tuple):
                    c = tuple(c)
                ok = UNDEF not in c
            else:
                c = toColor(v, UNDEF)
                ok = c is not UNDEF
            if ok:
                assigned[k] = c
                del kw[k]
                progress = 1
    if kw:
        raise ValueError("Can't convert\n%s" % str(kw))
    getAllNamedColors()
    for k, c in assigned.items():
        globals()[k] = c
        if isinstance(c, Color):
            _namedColors[k] = c