from collections import defaultdict
from typing import (
import numpy as np
from .errors import Errors
from .morphology import Morphology
from .tokens import Doc, Span, Token
from .training import Example
from .util import SimpleFrozenList, get_lang_class
@property
def fscore(self) -> float:
    p = self.precision
    r = self.recall
    return 2 * (p * r / (p + r + 1e-100))