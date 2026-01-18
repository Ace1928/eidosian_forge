import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, cast
import networkx as nx
import numpy as np
import pytest
import cirq
from cirq.experiments import (
from cirq.experiments.random_quantum_circuit_generation import (
def test_grid_interaction_layer_repr():
    layer = GridInteractionLayer(col_offset=0, vertical=True, stagger=False)
    assert repr(layer) == 'cirq.experiments.GridInteractionLayer(col_offset=0, vertical=True, stagger=False)'