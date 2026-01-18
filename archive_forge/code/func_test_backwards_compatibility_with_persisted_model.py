import os
import logging
import unittest
import numpy as np
from copy import deepcopy
import pytest
from gensim.models import EnsembleLda, LdaMulticore, LdaModel
from gensim.test.utils import datapath, get_tmpfile, common_corpus, common_dictionary
def test_backwards_compatibility_with_persisted_model(self):
    elda = self.get_elda()
    loaded_elda = EnsembleLda.load(datapath('ensemblelda'))
    np.testing.assert_allclose(elda.ttda, loaded_elda.ttda, rtol=RTOL)
    atol = loaded_elda.asymmetric_distance_matrix.max() * 1e-05
    np.testing.assert_allclose(elda.asymmetric_distance_matrix, loaded_elda.asymmetric_distance_matrix, atol=atol)