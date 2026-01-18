from typing import Tuple, List
from unittest.mock import create_autospec
import cirq
import numpy as np
from pyquil.gates import MEASURE, RX, DECLARE, H, CNOT, I
from pyquil.quilbase import Pragma, Reset
from cirq_rigetti import circuit_transformers as transformers
def reset_hook(program, measurement_id_map):
    program._instructions.insert(0, Reset())
    return (program, measurement_id_map)