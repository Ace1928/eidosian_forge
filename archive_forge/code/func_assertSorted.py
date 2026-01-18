import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def assertSorted(self, expected, parent_map):
    graph = self.make_known_graph(parent_map)
    value = graph.gc_sort()
    if expected != value:
        self.assertEqualDiff(pprint.pformat(expected), pprint.pformat(value))