from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_heads_limits_search_common_search_must_continue(self):
    graph_dict = {b'h1': [b'shortcut', b'common1'], b'h2': [b'common1'], b'shortcut': [b'common2'], b'common1': [b'common2'], b'common2': [b'deeper']}
    self.assertEqual({b'h1', b'h2'}, self._run_heads_break_deeper(graph_dict, [b'h1', b'h2']))