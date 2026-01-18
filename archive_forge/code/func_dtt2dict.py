import sys
from contextlib import nullcontext
import numpy as np
import pytest
from packaging.version import Version
from ..casting import (
from ..testing import suppress_warnings
def dtt2dict(dtt):
    """Create info dictionary from numpy type"""
    info = np.finfo(dtt)
    return dict(min=info.min, max=info.max, nexp=info.nexp, nmant=info.nmant, minexp=info.minexp, maxexp=info.maxexp, width=np.dtype(dtt).itemsize)