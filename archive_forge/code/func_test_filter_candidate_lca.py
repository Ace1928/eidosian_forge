from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_filter_candidate_lca(self):
    """Test filter_candidate_lca for a corner case

        This tests the case where we encounter the end of iteration for b'e'
        in the same pass as we discover that b'd' is an ancestor of b'e', and
        therefore b'e' can't be an lca.

        To compensate for different dict orderings on other Python
        implementations, we mirror b'd' and b'e' with b'b' and b'a'.
        """
    graph = self.make_graph({b'c': [b'b', b'd'], b'd': [b'e'], b'b': [b'a'], b'a': [NULL_REVISION], b'e': [NULL_REVISION]})
    self.assertEqual({b'c'}, graph.heads([b'a', b'c', b'e']))