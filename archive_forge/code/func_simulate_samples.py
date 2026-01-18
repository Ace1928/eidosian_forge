import json
from typing import Any, cast, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import networkx as nx
import numpy as np
import cirq
from cirq_aqt import aqt_device_metadata
def simulate_samples(self, repetitions: int) -> cirq.Result:
    """Samples the circuit.

        Args:
            repetitions: Number of times the circuit is simulated.

        Returns:
            Result from Cirq.Simulator.

        Raises:
            RuntimeError: Simulate called without a circuit.
        """
    if self.simulate_ideal:
        noise_model = cirq.NO_NOISE
    else:
        noise_model = AQTNoiseModel()
    if self.circuit == cirq.Circuit():
        raise RuntimeError('Simulate called without a valid circuit.')
    sim = cirq.DensityMatrixSimulator(noise=noise_model)
    result = sim.run(self.circuit, repetitions=repetitions)
    return result