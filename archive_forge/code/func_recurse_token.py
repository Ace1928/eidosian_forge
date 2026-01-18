from itertools import groupby
import numpy as np
import param
import pyparsing as pp
from ..core.options import Cycle, Options, Palette
from ..core.util import merge_option_dicts
from ..operation import Compositor
from .transform import dim
@classmethod
def recurse_token(cls, token, inner):
    recursed = []
    for tok in token:
        if isinstance(tok, list):
            new_tok = [s for t in tok for s in (cls.recurse_token(t, inner) if isinstance(t, list) else [t])]
            recursed.append(inner % ''.join(new_tok))
        else:
            recursed.append(tok)
    return inner % ''.join(recursed)