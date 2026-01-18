import itertools
import six
import numpy as np
from patsy import PatsyError
from patsy.categorical import (guess_categorical,
from patsy.util import (atleast_2d_column_default,
from patsy.design_info import (DesignMatrix, DesignInfo,
from patsy.redundancy import pick_contrasts_for_term
from patsy.eval import EvalEnvironment
from patsy.contrasts import code_contrast_matrix, Treatment
from patsy.compat import OrderedDict
from patsy.missing import NAAction
def test__factors_memorize():

    class MockFactor(object):

        def __init__(self, requested_passes, token):
            self._requested_passes = requested_passes
            self._token = token
            self._chunk_in_pass = 0
            self._seen_passes = 0

        def memorize_passes_needed(self, state, eval_env):
            state['calls'] = []
            state['token'] = self._token
            return self._requested_passes

        def memorize_chunk(self, state, which_pass, data):
            state['calls'].append(('memorize_chunk', which_pass))
            assert data['chunk'] == self._chunk_in_pass
            self._chunk_in_pass += 1

        def memorize_finish(self, state, which_pass):
            state['calls'].append(('memorize_finish', which_pass))
            self._chunk_in_pass = 0

    class Data(object):
        CHUNKS = 3

        def __init__(self):
            self.calls = 0
            self.data = [{'chunk': i} for i in range(self.CHUNKS)]

        def __call__(self):
            self.calls += 1
            return iter(self.data)
    data = Data()
    f0 = MockFactor(0, 'f0')
    f1 = MockFactor(1, 'f1')
    f2a = MockFactor(2, 'f2a')
    f2b = MockFactor(2, 'f2b')
    factor_states = _factors_memorize(set([f0, f1, f2a, f2b]), data, {})
    assert data.calls == 2
    mem_chunks0 = [('memorize_chunk', 0)] * data.CHUNKS
    mem_chunks1 = [('memorize_chunk', 1)] * data.CHUNKS
    expected = {f0: {'calls': [], 'token': 'f0'}, f1: {'calls': mem_chunks0 + [('memorize_finish', 0)], 'token': 'f1'}, f2a: {'calls': mem_chunks0 + [('memorize_finish', 0)] + mem_chunks1 + [('memorize_finish', 1)], 'token': 'f2a'}, f2b: {'calls': mem_chunks0 + [('memorize_finish', 0)] + mem_chunks1 + [('memorize_finish', 1)], 'token': 'f2b'}}
    assert factor_states == expected