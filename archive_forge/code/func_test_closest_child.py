import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_closest_child(self):
    """Test closest_child returns expected value and returns None for lowest node in hierarchy."""
    self.assertEqual(self.vectors.closest_child('dog.n.01'), 'terrier.n.01')
    self.assertEqual(self.vectors.closest_child('harbor_porpoise.n.01'), None)