from typing import Callable, List, Optional, Tuple, Set, Any, TYPE_CHECKING
import numpy as np
import cirq
from cirq_google.line.placement import place_strategy, optimization
from cirq_google.line.placement.chip import above, right_of, chip_as_adjacency_list, EDGE
from cirq_google.line.placement.sequence import GridQubitLineTuple, LineSequence
def index_2d(seqs: List[List[Any]], target: Any) -> Tuple[int, int]:
    """Finds the first index of a target item within a list of lists.

    Args:
        seqs: The list of lists to search.
        target: The item to find.

    Raises:
        ValueError: Item is not present.
    """
    for i in range(len(seqs)):
        for j in range(len(seqs[i])):
            if seqs[i][j] == target:
                return (i, j)
    raise ValueError('Item not present.')