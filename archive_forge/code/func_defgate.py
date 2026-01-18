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
def defgate(self, name: str, matrix: Union[List[List[Any]], np.ndarray, np.matrix], parameters: Optional[List[Parameter]]=None) -> 'Program':
    """
        Define a new static gate.

        .. note::

            The matrix elements along each axis are ordered by bitstring. For two qubits the order
            is ``00, 01, 10, 11``, where the the bits **are ordered in reverse** by the qubit index,
            i.e., for qubits 0 and 1 the bitstring ``01`` indicates that qubit 0 is in the state 1.
            See also :ref:`the related docs in the WavefunctionSimulator Overview <basis_ordering>`.


        :param name: The name of the gate.
        :param matrix: List of lists or Numpy 2d array.
        :param parameters: list of parameters that are used in this gate
        :return: The Program instance.
        """
    return self.inst(DefGate(name, matrix, parameters))