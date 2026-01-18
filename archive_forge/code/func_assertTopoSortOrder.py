import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def assertTopoSortOrder(self, ancestry):
    """Check topo_sort and iter_topo_order is genuinely topological order.

        For every child in the graph, check if it comes after all of it's
        parents.
        """
    graph = self.make_known_graph(ancestry)
    sort_result = graph.topo_sort()
    self.assertEqual(len(ancestry), len(sort_result))
    node_idx = {node: idx for idx, node in enumerate(sort_result)}
    for node in sort_result:
        parents = ancestry[node]
        for parent in parents:
            if parent not in ancestry:
                continue
            if node_idx[node] <= node_idx[parent]:
                self.fail('parent %s must come before child %s:\n%s' % (parent, node, sort_result))