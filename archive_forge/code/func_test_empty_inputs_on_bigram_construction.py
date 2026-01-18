import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
def test_empty_inputs_on_bigram_construction(self):
    """Test that empty inputs don't throw errors and return the expected result."""
    self.assertEqual(list(self.bigram_default[[]]), [])
    self.assertEqual(list(self.bigram_default[iter(())]), [])
    self.assertEqual(list(self.bigram_default[[[], []]]), [[], []])
    self.assertEqual(list(self.bigram_default[iter([[], []])]), [[], []])
    self.assertEqual(list(self.bigram_default[(iter(()) for i in range(2))]), [[], []])