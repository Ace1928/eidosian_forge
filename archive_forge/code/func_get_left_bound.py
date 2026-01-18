from typing import Iterator, Tuple, Union
from ...errors import Errors
from ...symbols import AUX, NOUN, PRON, PROPN, VERB
from ...tokens import Doc, Span
def get_left_bound(root):
    left_bound = root
    for tok in reversed(list(root.lefts)):
        if tok.dep in np_left_deps:
            left_bound = tok
    return left_bound