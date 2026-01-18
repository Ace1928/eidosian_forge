import logging
import unittest
import os
import bz2
import sys
import tempfile
import subprocess
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import word2vec, keyedvectors
from gensim.utils import check_output
from gensim.test.utils import (
@unittest.skipIf('BULK_TEST_REPS' not in os.environ, reason='bulk test only occasionally run locally')
def test_method_in_bulk(self):
    """Not run by default testing, but can be run locally to help tune stochastic aspects of tests
        to very-very-rarely fail. EG:
        % BULK_TEST_REPS=200 METHOD_NAME=test_cbow_hs pytest test_word2vec.py -k "test_method_in_bulk"
        Method must accept `ranks` keyword-argument, empty list into which salient internal result can be reported.
        """
    failures = 0
    ranks = []
    reps = int(os.environ['BULK_TEST_REPS'])
    method_name = os.environ.get('METHOD_NAME', 'test_cbow_hs')
    method_fn = getattr(self, method_name)
    for i in range(reps):
        try:
            method_fn(ranks=ranks)
        except Exception as ex:
            print('%s failed: %s' % (method_name, ex))
            failures += 1
    print(ranks)
    print(np.mean(ranks))
    self.assertEquals(failures, 0, 'too many failures')