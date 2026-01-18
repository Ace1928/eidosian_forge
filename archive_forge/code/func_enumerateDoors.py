from pyparsing import *
import random
import string
def enumerateDoors(l):
    if len(l) == 0:
        return ''
    out = []
    if len(l) > 1:
        out.append(', '.join(l[:-1]))
        out.append('and')
    out.append(l[-1])
    return ' '.join(out)