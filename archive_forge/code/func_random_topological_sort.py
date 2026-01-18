import operator
import random
from typing import Any, Callable, cast, Iterable, TYPE_CHECKING
import networkx
from cirq import ops
def random_topological_sort(dag: networkx.DiGraph) -> Iterable[Any]:
    remaining_dag = dag.copy()
    frontier = list((node for node in remaining_dag.nodes() if not remaining_dag.pred[node]))
    while frontier:
        random.shuffle(frontier)
        node = frontier.pop()
        succ = remaining_dag.succ[node]
        remaining_dag.remove_node(node)
        frontier.extend((new_node for new_node in succ if not remaining_dag.pred[new_node]))
        yield node
    assert not remaining_dag