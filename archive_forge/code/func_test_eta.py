import logging
import numbers
import os
import unittest
import copy
import numpy as np
from numpy.testing import assert_allclose
from gensim.corpora import mmcorpus, Dictionary
from gensim.models import ldamodel, ldamulticore
from gensim import matutils, utils
from gensim.test import basetmtests
from gensim.test.utils import datapath, get_tmpfile, common_texts
def test_eta(self):
    kwargs = dict(id2word=dictionary, num_topics=2, eta=None)
    num_terms = len(dictionary)
    expected_shape = (num_terms,)
    model = self.class_(**kwargs)
    self.assertEqual(model.eta.shape, expected_shape)
    assert_allclose(model.eta, np.array([0.5] * num_terms))
    kwargs['eta'] = 'symmetric'
    model = self.class_(**kwargs)
    self.assertEqual(model.eta.shape, expected_shape)
    assert_allclose(model.eta, np.array([0.5] * num_terms))
    kwargs['eta'] = 0.3
    model = self.class_(**kwargs)
    self.assertEqual(model.eta.shape, expected_shape)
    assert_allclose(model.eta, np.array([0.3] * num_terms))
    kwargs['eta'] = 3
    model = self.class_(**kwargs)
    self.assertEqual(model.eta.shape, expected_shape)
    assert_allclose(model.eta, np.array([3] * num_terms))
    kwargs['eta'] = [0.3] * num_terms
    model = self.class_(**kwargs)
    self.assertEqual(model.eta.shape, expected_shape)
    assert_allclose(model.eta, np.array([0.3] * num_terms))
    kwargs['eta'] = np.array([0.3] * num_terms)
    model = self.class_(**kwargs)
    self.assertEqual(model.eta.shape, expected_shape)
    assert_allclose(model.eta, np.array([0.3] * num_terms))
    testeta = np.array([[0.5] * len(dictionary)] * 2)
    kwargs['eta'] = testeta
    self.class_(**kwargs)
    kwargs['eta'] = testeta.reshape(tuple(reversed(testeta.shape)))
    self.assertRaises(AssertionError, self.class_, **kwargs)
    kwargs['eta'] = [0.3]
    self.assertRaises(AssertionError, self.class_, **kwargs)
    kwargs['eta'] = [0.3] * (num_terms + 1)
    self.assertRaises(AssertionError, self.class_, **kwargs)
    kwargs['eta'] = 'gensim is cool'
    self.assertRaises(ValueError, self.class_, **kwargs)
    kwargs['eta'] = 'asymmetric'
    self.assertRaises(ValueError, self.class_, **kwargs)