import unittest
from collections import defaultdict
from nltk.translate import AlignedSent, IBMModel
from nltk.translate.ibm_model import AlignmentInfo
def prob_t_a_given_s_mock(a):
    prob_values = {(0, 3, 2): 0.5, (0, 2, 2): 0.6, (0, 1, 1): 0.4, (0, 3, 3): 0.6, (0, 4, 4): 0.7}
    return prob_values.get(a.alignment, 0.01)