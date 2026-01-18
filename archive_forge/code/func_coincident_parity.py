import re
from itertools import product
import numpy as np
import copy
from typing import (
from pyquil.quilatom import (
from .quil import Program
from .gates import H, RZ, RX, CNOT, X, PHASE, QUANTUM_GATES
from numbers import Number, Complex
from collections import OrderedDict
import warnings
def coincident_parity(p1: PauliTerm, p2: PauliTerm) -> bool:
    non_similar = 0
    p1_indices = set(p1._ops.keys())
    p2_indices = set(p2._ops.keys())
    for idx in p1_indices.intersection(p2_indices):
        if p1[idx] != p2[idx]:
            non_similar += 1
    return non_similar % 2 == 0