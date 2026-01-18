from collections import defaultdict
from typing import (
import numpy as np
from .errors import Errors
from .morphology import Morphology
from .tokens import Doc, Span, Token
from .training import Example
from .util import SimpleFrozenList, get_lang_class
def score_set(self, cand, gold) -> None:
    self.cands.append(cand)
    self.golds.append(gold)