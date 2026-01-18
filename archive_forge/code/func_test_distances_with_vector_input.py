import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_distances_with_vector_input(self):
    """Test that distances between input vector and a list of words have expected values."""
    input_vector = self.vectors['dog.n.01']
    distances = self.vectors.distances(input_vector, ['mammal.n.01', 'dog.n.01'])
    self.assertTrue(np.allclose(distances, [4.5278745, 0]))
    distances = self.vectors.distances(input_vector)
    self.assertEqual(len(distances), len(self.vectors))
    self.assertTrue(np.allclose(distances[-1], 10.04756))