from typing import Tuple, Any, List, Union, Dict
import pytest
import cirq
from pyquil import Program
import numpy as np
import sympy
from cirq_rigetti import circuit_sweep_executors as executors, circuit_transformers
def broken_hook(program: Program, measurement_id_map: Dict[str, str]) -> Tuple[Program, Dict[str, str]]:
    return (program, {cirq_key: f'{cirq_key}-doesnt-exist' for cirq_key in measurement_id_map})