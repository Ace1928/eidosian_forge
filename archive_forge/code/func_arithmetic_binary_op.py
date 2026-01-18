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
def arithmetic_binary_op(self, op, left, right):
    if op == 'ADD':
        return ClassicalAdd(left, right)
    elif op == 'SUB':
        return ClassicalSub(left, right)
    elif op == 'MUL':
        return ClassicalMul(left, right)
    elif op == 'DIV':
        return ClassicalDiv(left, right)