from __future__ import annotations
from typing import Any
from typing import Collection
from typing import DefaultDict
from typing import Iterable
from typing import Iterator
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TypeVar
from .. import util
from ..exc import CircularDependencyError
def sort_as_subsets(tuples: Collection[Tuple[_T, _T]], allitems: Collection[_T]) -> Iterator[Sequence[_T]]:
    edges: DefaultDict[_T, Set[_T]] = util.defaultdict(set)
    for parent, child in tuples:
        edges[child].add(parent)
    todo = list(allitems)
    todo_set = set(allitems)
    while todo_set:
        output = []
        for node in todo:
            if todo_set.isdisjoint(edges[node]):
                output.append(node)
        if not output:
            raise CircularDependencyError('Circular dependency detected.', find_cycles(tuples, allitems), _gen_edges(edges))
        todo_set.difference_update(output)
        todo = [t for t in todo if t in todo_set]
        yield output