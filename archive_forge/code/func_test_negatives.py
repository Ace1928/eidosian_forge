import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_negatives(self):
    """Tests that correct number of negatives are sampled."""
    model = PoincareModel(self.data, negative=5)
    self.assertEqual(len(model._get_candidate_negatives()), 5)