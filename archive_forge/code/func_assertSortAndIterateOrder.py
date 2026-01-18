import pprint
from breezy.errors import GraphCycleError
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase
from breezy.tsort import MergeSorter, TopoSorter, merge_sort, topo_sort
def assertSortAndIterateOrder(self, graph):
    """Check topo_sort and iter_topo_order is genuinely topological order.

        For every child in the graph, check if it comes after all of it's
        parents.
        """
    sort_result = topo_sort(graph)
    iter_result = list(TopoSorter(graph).iter_topo_order())
    for node, parents in graph:
        for parent in parents:
            if sort_result.index(node) < sort_result.index(parent):
                self.fail('parent %s must come before child %s:\n%s' % (parent, node, sort_result))
            if iter_result.index(node) < iter_result.index(parent):
                self.fail('parent %s must come before child %s:\n%s' % (parent, node, iter_result))