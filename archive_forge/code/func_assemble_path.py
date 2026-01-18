from typing import Dict, List, Optional, Set, TYPE_CHECKING
import abc
import collections
from cirq.devices import GridQubit
from cirq_google.line.placement import place_strategy
from cirq_google.line.placement.chip import chip_as_adjacency_list
from cirq_google.line.placement.sequence import GridQubitLineTuple
def assemble_path(n: GridQubit, parent: Dict[GridQubit, GridQubit]):
    path = [n]
    while n in parent:
        n = parent[n]
        path.append(n)
    return path