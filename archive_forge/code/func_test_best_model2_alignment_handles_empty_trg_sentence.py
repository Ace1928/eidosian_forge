import unittest
from collections import defaultdict
from nltk.translate import AlignedSent, IBMModel
from nltk.translate.ibm_model import AlignmentInfo
def test_best_model2_alignment_handles_empty_trg_sentence(self):
    sentence_pair = AlignedSent([], TestIBMModel.__TEST_SRC_SENTENCE)
    ibm_model = IBMModel([])
    a_info = ibm_model.best_model2_alignment(sentence_pair)
    self.assertEqual(a_info.alignment[1:], ())
    self.assertEqual(a_info.cepts, [[], [], [], [], []])