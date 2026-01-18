import itertools
import types
import warnings
from collections import defaultdict
from typing import (
import numpy as np
from rpcq.messages import NativeQuilMetadata, ParameterAref
from pyquil._parser.parser import run_parser
from pyquil._memory import Memory
from pyquil.gates import MEASURE, RESET, MOVE
from pyquil.noise import _check_kraus_ops, _create_kraus_pragmas, pauli_kraus_map
from pyquil.quilatom import (
from pyquil.quilbase import (
from pyquil.quiltcalibrations import (
def get_classical_addresses_from_program(program: Program) -> Dict[str, List[int]]:
    """
    Returns a sorted list of classical addresses found in the MEASURE instructions in the program.

    :param program: The program from which to get the classical addresses.
    :return: A mapping from memory region names to lists of offsets appearing in the program.
    """
    addresses: Dict[str, List[int]] = defaultdict(list)
    flattened_addresses = {}
    for instr in program:
        if isinstance(instr, Measurement) and instr.classical_reg:
            addresses[instr.classical_reg.name].append(instr.classical_reg.offset)
    for k, v in addresses.items():
        reduced_list = list(set(v))
        reduced_list.sort()
        flattened_addresses[k] = reduced_list
    return flattened_addresses