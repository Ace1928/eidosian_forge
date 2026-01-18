import collections
import enum
from taskflow.engines.action_engine import compiler as co
def breadth_first_iterate(execution_graph, starting_node, direction, through_flows=True, through_retries=True, through_tasks=True):
    """Iterates connected nodes in execution graph (from starting node).

    Does so in a breadth first manner.

    Jumps over nodes with ``noop`` attribute (does not yield them back).
    """
    initial_nodes_iter, connected_to_functors = _extract_connectors(execution_graph, starting_node, direction, through_flows=through_flows, through_retries=through_retries, through_tasks=through_tasks)
    q = collections.deque(initial_nodes_iter)
    while q:
        node = q.popleft()
        node_attrs = execution_graph.nodes[node]
        if not node_attrs.get('noop'):
            yield node
        try:
            node_kind = node_attrs['kind']
            connected_to_functor = connected_to_functors[node_kind]
        except KeyError:
            pass
        else:
            q.extend(connected_to_functor(node))