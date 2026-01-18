import unittest
from collections import defaultdict
from nltk.translate import AlignedSent, IBMModel
from nltk.translate.ibm_model import AlignmentInfo
def test_best_model2_alignment_handles_empty_src_sentence(self):
    sentence_pair = AlignedSent(TestIBMModel.__TEST_TRG_SENTENCE, [])
    ibm_model = IBMModel([])
    a_info = ibm_model.best_model2_alignment(sentence_pair)
    self.assertEqual(a_info.alignment[1:], (0, 0, 0))
    self.assertEqual(a_info.cepts, [[1, 2, 3]])