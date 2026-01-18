import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def module_scenarios():
    scenarios = [('python', {'_gc_module': _groupcompress_py})]
    if compiled_groupcompress_feature.available():
        gc_module = compiled_groupcompress_feature.module
        scenarios.append(('C', {'_gc_module': gc_module}))
    return scenarios