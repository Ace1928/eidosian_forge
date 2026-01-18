from typing import Any, Callable, Dict, Generic, Iterator, TypeVar, cast, TYPE_CHECKING
import functools
import networkx
from cirq import ops
from cirq.circuits import circuit
def all_operations(self) -> Iterator['cirq.Operation']:
    return (node.val for node in self.ordered_nodes())