from typing import Dict, List, Optional, Set, TYPE_CHECKING
import abc
import collections
from cirq.devices import GridQubit
from cirq_google.line.placement import place_strategy
from cirq_google.line.placement.chip import chip_as_adjacency_list
from cirq_google.line.placement.sequence import GridQubitLineTuple
def get_or_search(self) -> List[GridQubit]:
    """Starts the search or gives previously calculated sequence.

        Returns:
            The linear qubit sequence found.
        """
    if not self._sequence:
        self._sequence = self._find_sequence()
    return self._sequence