from dataclasses import dataclass
from typing import Optional, List, Callable, Dict, Tuple, Set, Any
import networkx as nx
import numpy as np
import pandas as pd
import cirq
import cirq.contrib.routing as ccr
def replace_swap_permutation_gate(op: 'cirq.Operation', _):
    if isinstance(op.gate, cirq.contrib.acquaintance.SwapPermutationGate):
        return [op.gate.swap_gate.on(*op.qubits)]
    return op