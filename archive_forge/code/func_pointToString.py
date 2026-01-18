from typing import Callable
from fontTools.pens.basePen import BasePen
def pointToString(pt, ntos=str):
    return ' '.join((ntos(i) for i in pt))