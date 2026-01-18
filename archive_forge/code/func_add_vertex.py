import itertools
import random
from typing import Any, Dict, FrozenSet, Hashable, Iterable, Mapping, Optional, Set, Tuple, Union
def add_vertex(self, vertex: Hashable) -> None:
    if vertex not in self._adjacency_lists:
        self._adjacency_lists[vertex] = set()