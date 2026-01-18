import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_data_counts_with_bytes(self):
    """Tests whether input bytes data is loaded correctly and completely."""
    model = PoincareModel([(b'\x80\x01c', b'Pqa'), (b'node.1', b'node.2')])
    self.assertEqual(len(model.all_relations), 2)
    self.assertEqual(len(model.node_relations[model.kv.get_index(b'\x80\x01c')]), 1)
    self.assertEqual(len(model.kv), 4)
    self.assertTrue(b'Pqa' not in model.node_relations)