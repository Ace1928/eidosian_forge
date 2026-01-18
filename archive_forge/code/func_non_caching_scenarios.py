import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def non_caching_scenarios():
    scenarios = [('python-nocache', {'module': _known_graph_py, 'do_cache': False})]
    if compiled_known_graph_feature.available():
        scenarios.append(('C-nocache', {'module': compiled_known_graph_feature.module, 'do_cache': False}))
    return scenarios