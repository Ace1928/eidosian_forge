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
def find_cycles(tuples: Iterable[Tuple[_T, _T]], allitems: Iterable[_T]) -> Set[_T]:
    edges: DefaultDict[_T, Set[_T]] = util.defaultdict(set)
    for parent, child in tuples:
        edges[parent].add(child)
    nodes_to_test = set(edges)
    output = set()
    for node in nodes_to_test:
        stack = [node]
        todo = nodes_to_test.difference(stack)
        while stack:
            top = stack[-1]
            for node in edges[top]:
                if node in stack:
                    cyc = stack[stack.index(node):]
                    todo.difference_update(cyc)
                    output.update(cyc)
                if node in todo:
                    stack.append(node)
                    todo.remove(node)
                    break
            else:
                node = stack.pop()
    return output