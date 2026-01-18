from typing import (
import numpy as np
import cirq
from cirq import value
from cirq_google.calibration.phased_fsim import (
def run_sweep_iter(self, program: cirq.AbstractCircuit, params: cirq.Sweepable, repetitions: int=1) -> Iterator[cirq.Result]:
    converted = _convert_to_circuit_with_drift(self, program)
    yield from self._simulator.run_sweep_iter(converted, params, repetitions)