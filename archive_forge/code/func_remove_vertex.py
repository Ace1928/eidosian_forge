import itertools
import random
from typing import Any, Dict, FrozenSet, Hashable, Iterable, Mapping, Optional, Set, Tuple, Union
def remove_vertex(self, vertex: Hashable) -> None:
    for edge in self._adjacency_lists[vertex]:
        del self._labelled_edges[edge]
        for neighbor in edge.difference((vertex,)):
            self._adjacency_lists[neighbor].difference_update((edge,))
    del self._adjacency_lists[vertex]