import os
import re
import random
from gimpfu import *
def makepatterns(allow, include=None, exclude=None):
    src = set()
    src.update([x for x in allow])
    src.update([allow[:i] for i in range(1, len(allow) + 1)])
    for i in range(len(allow)):
        pick1, pick2 = (random.choice(allow), random.choice(allow))
        src.update([pick1 + pick2])
    for i in range(3, 11) + range(14, 18) + range(31, 34):
        src.update([''.join([random.choice(allow) for k in range(i)])])
    out = []
    for srcpat in src:
        if exclude and exclude in srcpat:
            continue
        if include and include not in srcpat:
            out.append(include + srcpat[1:])
            continue
        out.append(srcpat)
    return list(set(out))