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
def pauli_term(self, name, expression, qubits):
    from pyquil.paulis import PauliTerm
    return PauliTerm.from_list(list(zip(name, qubits)), expression)