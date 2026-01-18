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
def is_supported_on_qpu(self) -> bool:
    """
        Whether the program can be compiled to the hardware to execute on a QPU. These Quil
        programs are more restricted than Protoquil: for instance, RESET must be before any
        gates or MEASUREs, and MEASURE on a qubit must be after any gates on that qubit.

        :return: True if the Program is supported Quil, False otherwise
        """
    try:
        validate_supported_quil(self)
        return True
    except ValueError:
        return False