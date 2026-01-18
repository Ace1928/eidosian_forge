from typing import Any, Callable, Dict, Generic, Iterator, TypeVar, cast, TYPE_CHECKING
import functools
import networkx
from cirq import ops
from cirq.circuits import circuit
def get_next_node(succ: networkx.classes.coreviews.AtlasView) -> Unique['cirq.Operation']:
    if succ:
        return get_root_node(next(iter(succ)))
    return get_first_node()