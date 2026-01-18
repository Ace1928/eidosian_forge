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
def reverse_hack(p: Program) -> Program:
    revp = Program()
    revp.inst(list(reversed(p.instructions)))
    return revp