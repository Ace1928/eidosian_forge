import itertools
from typing import Iterable, Sequence, Tuple, TypeVar, TYPE_CHECKING
from cirq import circuits, ops
from cirq.contrib.acquaintance.gates import acquaint
from cirq.contrib.acquaintance.permutation import LinearPermutationGate, SwapPermutationGate
def skip_and_wrap_around(items: Sequence[TItem]) -> Tuple[TItem, ...]:
    n_items = len(items)
    positions = {p: i for i, p in enumerate(itertools.chain(range(0, n_items, 2), reversed(range(1, n_items, 2))))}
    return tuple((items[positions[i]] for i in range(n_items)))