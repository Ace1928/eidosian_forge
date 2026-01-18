from collections import namedtuple
import unittest
import logging
import numpy as np
import pytest
from scipy.spatial.distance import cosine
from gensim.models.doc2vec import Doc2Vec
from gensim import utils
from gensim.models import translation_matrix
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
@unittest.skip('flaky test likely to be discarded when <https://github.com/RaRe-Technologies/gensim/issues/2977> is addressed')
def test_infer_vector(self):
    """Test that translation gives similar results to traditional inference.

        This may not be completely sensible/salient with such tiny data, but
        replaces what seemed to me to be an ever-more-nonsensical test.

        See <https://github.com/RaRe-Technologies/gensim/issues/2977> for discussion
        of whether the class this supposedly tested even survives when the
        TranslationMatrix functionality is better documented.
        """
    model = translation_matrix.BackMappingTranslationMatrix(self.source_doc_vec, self.target_doc_vec, self.train_docs[:5])
    model.train(self.train_docs[:5])
    backmapped_vec = model.infer_vector(self.target_doc_vec.dv[self.train_docs[5].tags[0]])
    self.assertEqual(backmapped_vec.shape, (8,))
    d2v_inferred_vector = self.source_doc_vec.infer_vector(self.train_docs[5].words)
    distance = cosine(backmapped_vec, d2v_inferred_vector)
    self.assertLessEqual(distance, 0.1)