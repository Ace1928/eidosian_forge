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
def def_circuit(self, name, variables, qubits, instrs):
    qubits = qubits if qubits else []
    space = ' ' if qubits else ''
    if variables:
        raw_defcircuit = 'DEFCIRCUIT {}({}){}{}:'.format(name, ', '.join(map(str, variables)), space, ' '.join(map(str, qubits)))
    else:
        raw_defcircuit = 'DEFCIRCUIT {}{}{}:'.format(name, space, ' '.join(map(str, qubits)))
    raw_defcircuit += '\n    '.join([''] + [str(instr) for instr in instrs])
    return RawInstr(raw_defcircuit)