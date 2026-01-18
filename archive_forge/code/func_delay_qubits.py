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
def delay_qubits(self, qubits, delay_amount=None):
    if delay_amount is None:
        delay_amount = int(qubits[-1].index)
        qubits = qubits[:-1]
    d = DELAY(*[*qubits, delay_amount])
    return d