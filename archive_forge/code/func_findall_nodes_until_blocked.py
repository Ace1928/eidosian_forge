from typing import Any, Callable, Dict, Generic, Iterator, TypeVar, cast, TYPE_CHECKING
import functools
import networkx
from cirq import ops
from cirq.circuits import circuit
def findall_nodes_until_blocked(self, is_blocker: Callable[['cirq.Operation'], bool]) -> Iterator[Unique['cirq.Operation']]:
    """Finds all nodes before blocking ones.

        Args:
            is_blocker: The predicate that indicates whether or not an
            operation is blocking.
        """
    remaining_dag = self.copy()
    for node in self.ordered_nodes():
        if node not in remaining_dag:
            continue
        if is_blocker(node.val):
            successors = list(remaining_dag.succ[node])
            remaining_dag.remove_nodes_from(successors)
            remaining_dag.remove_node(node)
            continue
        yield node