from typing import Iterator, Tuple, Union
from ...errors import Errors
from ...symbols import NOUN, PRON, PROPN
from ...tokens import Doc, Span
def extend_right(w):
    rindex = w.i + 1
    for rdep in doc[w.i].rights:
        if rdep.dep == flat and rdep.pos in (NOUN, PROPN):
            rindex = rdep.i + 1
        else:
            break
    return rindex