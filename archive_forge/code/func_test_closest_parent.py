import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_closest_parent(self):
    """Test closest_parent returns expected value and returns None for highest node in hierarchy."""
    self.assertEqual(self.vectors.closest_parent('dog.n.01'), 'canine.n.02')
    self.assertEqual(self.vectors.closest_parent('mammal.n.01'), None)