from dataclasses import dataclass
from typing import Optional, List, Callable, Dict, Tuple, Set, Any
import networkx as nx
import numpy as np
import pandas as pd
import cirq
import cirq.contrib.routing as ccr
def compute_heavy_set(circuit: cirq.Circuit) -> List[int]:
    """Classically compute the heavy set of the given circuit.

    The heavy set is defined as the output bit-strings that have a greater than
    median probability of being generated.

    Args:
        circuit: The circuit to classically simulate.

    Returns:
        A list containing all of the heavy bit-string results.
    """
    simulator = cirq.Simulator()
    results = simulator.simulate(program=circuit)
    median = np.median(np.abs(results.state_vector() ** 2))
    return [idx for idx, amp in enumerate(results.state_vector()) if np.abs(amp ** 2) > median]