import logging
import unittest
from gensim.corpora import mmcorpus, Dictionary
from gensim.models import hdpmodel
from gensim.test import basetmtests
from gensim.test.utils import datapath, common_texts
import numpy as np
def test_ldamodel(self):
    """
        Create ldamodel object, and check if the corresponding alphas are equal.
        """
    ldam = self.model.suggested_lda_model()
    self.assertEqual(ldam.alpha[0], self.model.lda_alpha[0])