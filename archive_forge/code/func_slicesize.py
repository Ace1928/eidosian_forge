import itertools
import sys
import logging
from .formatstring import fmtstr
from .formatstring import normalize_slice
from .formatstring import FmtStr
from typing import (
def slicesize(s: slice) -> int:
    return int((s.stop - s.start) / (s.step if s.step else 1))