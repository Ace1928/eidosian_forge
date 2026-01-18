from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_heads_limits_search(self):
    graph_dict = {b'left': [b'common'], b'right': [b'common'], b'common': [b'deeper']}
    self.assertEqual({b'left', b'right'}, self._run_heads_break_deeper(graph_dict, [b'left', b'right']))