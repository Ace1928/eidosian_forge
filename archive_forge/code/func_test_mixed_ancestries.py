import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_mixed_ancestries(self):
    self.assertSorted([('F', 'c'), ('F', 'b'), ('F', 'a'), ('G', 'c'), ('G', 'b'), ('G', 'a'), ('Q', 'c'), ('Q', 'b'), ('Q', 'a')], {('F', 'a'): (), ('F', 'b'): (('F', 'a'),), ('F', 'c'): (('F', 'b'),), ('G', 'a'): (), ('G', 'b'): (('G', 'a'),), ('G', 'c'): (('G', 'b'),), ('Q', 'a'): (), ('Q', 'b'): (('Q', 'a'),), ('Q', 'c'): (('Q', 'b'),)})