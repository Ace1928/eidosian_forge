from typing import Any, Callable, Dict, Generic, Iterator, TypeVar, cast, TYPE_CHECKING
import functools
import networkx
from cirq import ops
from cirq.circuits import circuit
def get_first_node() -> Unique['cirq.Operation']:
    return get_root_node(next(iter(g.nodes())))