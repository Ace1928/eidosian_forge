import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def test_add_single(self):
    """Test that adding entity in a manual way works correctly."""
    entities = [f'___some_entity{i}_not_present_in_keyed_vectors___' for i in range(5)]
    vectors = [np.random.randn(self.vectors.vector_size) for _ in range(5)]
    for ent, vector in zip(entities, vectors):
        self.vectors.add_vectors(ent, vector)
    for ent, vector in zip(entities, vectors):
        self.assertTrue(np.allclose(self.vectors[ent], vector))
    kv = KeyedVectors(self.vectors.vector_size)
    for ent, vector in zip(entities, vectors):
        kv.add_vectors(ent, vector)
    for ent, vector in zip(entities, vectors):
        self.assertTrue(np.allclose(kv[ent], vector))