import os
import logging
import unittest
import numpy as np
from copy import deepcopy
import pytest
from gensim.models import EnsembleLda, LdaMulticore, LdaModel
from gensim.test.utils import datapath, get_tmpfile, common_corpus, common_dictionary
def test_generate_gensim_representation(self):
    elda = self.get_elda()
    gensim_model = elda.generate_gensim_representation()
    topics = gensim_model.get_topics()
    np.testing.assert_allclose(elda.get_topics(), topics, rtol=RTOL)