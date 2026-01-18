import unittest
from collections import defaultdict
from nltk.translate import AlignedSent, IBMModel
from nltk.translate.ibm_model import AlignmentInfo
def test_best_model2_alignment_does_not_change_pegged_alignment(self):
    sentence_pair = AlignedSent(TestIBMModel.__TEST_TRG_SENTENCE, TestIBMModel.__TEST_SRC_SENTENCE)
    translation_table = {'i': {"j'": 0.9, 'aime': 0.05, 'bien': 0.02, 'jambon': 0.03, None: 0}, 'love': {"j'": 0.05, 'aime': 0.9, 'bien': 0.01, 'jambon': 0.01, None: 0.03}, 'ham': {"j'": 0, 'aime': 0.01, 'bien': 0, 'jambon': 0.99, None: 0}}
    alignment_table = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.2))))
    ibm_model = IBMModel([])
    ibm_model.translation_table = translation_table
    ibm_model.alignment_table = alignment_table
    a_info = ibm_model.best_model2_alignment(sentence_pair, 2, 4)
    self.assertEqual(a_info.alignment[1:], (1, 4, 4))
    self.assertEqual(a_info.cepts, [[], [1], [], [], [2, 3]])