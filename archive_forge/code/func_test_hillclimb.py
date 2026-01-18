import unittest
from collections import defaultdict
from nltk.translate import AlignedSent, IBMModel
from nltk.translate.ibm_model import AlignmentInfo
def test_hillclimb(self):
    initial_alignment = AlignmentInfo((0, 3, 2), None, None, None)

    def neighboring_mock(a, j):
        if a.alignment == (0, 3, 2):
            return {AlignmentInfo((0, 2, 2), None, None, None), AlignmentInfo((0, 1, 1), None, None, None)}
        elif a.alignment == (0, 2, 2):
            return {AlignmentInfo((0, 3, 3), None, None, None), AlignmentInfo((0, 4, 4), None, None, None)}
        return set()

    def prob_t_a_given_s_mock(a):
        prob_values = {(0, 3, 2): 0.5, (0, 2, 2): 0.6, (0, 1, 1): 0.4, (0, 3, 3): 0.6, (0, 4, 4): 0.7}
        return prob_values.get(a.alignment, 0.01)
    ibm_model = IBMModel([])
    ibm_model.neighboring = neighboring_mock
    ibm_model.prob_t_a_given_s = prob_t_a_given_s_mock
    best_alignment = ibm_model.hillclimb(initial_alignment)
    self.assertEqual(best_alignment.alignment, (0, 4, 4))