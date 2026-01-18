from typing import Callable, List, Optional, Tuple, Set, Any, TYPE_CHECKING
import numpy as np
import cirq
from cirq_google.line.placement import place_strategy, optimization
from cirq_google.line.placement.chip import above, right_of, chip_as_adjacency_list, EDGE
from cirq_google.line.placement.sequence import GridQubitLineTuple, LineSequence
def assemble_edges() -> Set[EDGE]:
    """Creates list of edges for initial state.

            Returns:
              List of all possible edges.
            """
    nodes_set = set(self._c)
    edges = set()
    for n in self._c:
        if above(n) in nodes_set:
            edges.add(self._normalize_edge((n, above(n))))
        if right_of(n) in nodes_set:
            edges.add(self._normalize_edge((n, right_of(n))))
    return edges