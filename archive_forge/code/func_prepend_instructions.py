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
def prepend_instructions(self, instructions: Iterable[AbstractInstruction]) -> 'Program':
    """
        Prepend instructions to the beginning of the program.
        """
    self._instructions = [*instructions, *self._instructions]
    self._synthesized_instructions = None
    return self