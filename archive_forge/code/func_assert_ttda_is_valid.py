import os
import logging
import unittest
import numpy as np
from copy import deepcopy
import pytest
from gensim.models import EnsembleLda, LdaMulticore, LdaModel
from gensim.test.utils import datapath, get_tmpfile, common_corpus, common_dictionary
def assert_ttda_is_valid(self, elda):
    """Check that ttda has one or more topic and that term probabilities add to one."""
    assert len(elda.ttda) > 0
    sum_over_terms = elda.ttda.sum(axis=1)
    expected_sum_over_terms = np.ones(len(elda.ttda)).astype(np.float32)
    np.testing.assert_allclose(sum_over_terms, expected_sum_over_terms, rtol=0.0001)