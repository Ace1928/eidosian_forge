import json
import pkgutil
import operator
from typing import List
from deprecated import deprecated
from deprecated.sphinx import versionadded
from lark import Lark, Transformer, v_args
import numpy as np
from pyquil.quilbase import (
from pyquil.quiltwaveforms import _wf_from_dict
from pyquil.quilatom import (
from pyquil.gates import (
@v_args(inline=True)
def classical_comparison(self, op, target, left, right):
    if op == 'EQ':
        return ClassicalEqual(target, left, right)
    elif op == 'GT':
        return ClassicalGreaterThan(target, left, right)
    elif op == 'GE':
        return ClassicalGreaterEqual(target, left, right)
    elif op == 'LT':
        return ClassicalLessThan(target, left, right)
    elif op == 'LE':
        return ClassicalLessEqual(target, left, right)