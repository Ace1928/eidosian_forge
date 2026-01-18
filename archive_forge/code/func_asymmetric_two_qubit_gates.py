import dataclasses
from typing import Any, Dict, List, Sequence, Set, Type, TypeVar, Union
import numpy as np
import cirq, cirq_google
from cirq import _compat, devices
from cirq.devices import noise_utils
from cirq.transformers.heuristic_decompositions import gate_tabulation_math_utils
@classmethod
def asymmetric_two_qubit_gates(cls) -> Set[type]:
    return set()