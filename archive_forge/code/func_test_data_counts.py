import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_data_counts(self):
    """Tests whether data has been loaded correctly and completely."""
    model = PoincareModel(self.data)
    self.assertEqual(len(model.all_relations), 5)
    self.assertEqual(len(model.node_relations[model.kv.get_index('kangaroo.n.01')]), 3)
    self.assertEqual(len(model.kv), 7)
    self.assertTrue('mammal.n.01' not in model.node_relations)