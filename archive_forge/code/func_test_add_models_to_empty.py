import os
import logging
import unittest
import numpy as np
from copy import deepcopy
import pytest
from gensim.models import EnsembleLda, LdaMulticore, LdaModel
from gensim.test.utils import datapath, get_tmpfile, common_corpus, common_dictionary
def test_add_models_to_empty(self):
    elda = self.get_elda()
    ensemble = EnsembleLda(id2word=common_dictionary, num_models=0)
    ensemble.add_model(elda.ttda[0:1])
    ensemble.add_model(elda.ttda[1:])
    ensemble.recluster()
    np.testing.assert_allclose(ensemble.get_topics(), elda.get_topics(), rtol=RTOL)
    fname = get_tmpfile('gensim_models_ensemblelda')
    ensemble.save(fname)
    loaded_ensemble = EnsembleLda.load(fname)
    np.testing.assert_allclose(loaded_ensemble.get_topics(), elda.get_topics(), rtol=RTOL)
    self.test_inference(loaded_ensemble)