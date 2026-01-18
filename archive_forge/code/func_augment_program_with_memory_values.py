import warnings
from typing import Dict, List, Union, Optional, Set, cast, Iterable, Sequence, Tuple
import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._qvm import (
from pyquil.api._qvm_client import (
from pyquil.gates import MOVE
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program, percolate_declares
from pyquil.quilatom import MemoryReference
from pyquil.wavefunction import Wavefunction
@staticmethod
def augment_program_with_memory_values(quil_program: Program, memory_map: Dict[str, List[Union[int, float]]]) -> Program:
    p = Program()
    if len(memory_map.keys()) == 0:
        return quil_program
    elif isinstance(list(memory_map.keys())[0], str):
        for name, arr in memory_map.items():
            for index, value in enumerate(arr):
                p += MOVE(MemoryReference(name, offset=index), value)
    else:
        raise TypeError('Bad memory_map type; expected Dict[str, List[Union[int, float]]].')
    p += quil_program
    return percolate_declares(p)