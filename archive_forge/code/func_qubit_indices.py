from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence, Mapping, Optional, Set, Tuple, cast
from warnings import warn
from pyquil.quil import Program
from pyquil.quilatom import ParameterDesignator, QubitDesignator, format_parameter
from pyquil.quilbase import (
def qubit_indices(instr: AbstractInstruction) -> List[int]:
    """
    Get a list of indices associated with the given instruction.
    """
    if isinstance(instr, Measurement):
        return [instr.qubit.index]
    elif isinstance(instr, Gate):
        return [qubit.index for qubit in instr.qubits]
    else:
        return []