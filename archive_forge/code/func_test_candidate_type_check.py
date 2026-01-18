import unittest
from nltk.translate.meteor_score import meteor_score
def test_candidate_type_check(self):
    str_candidate = ' '.join(self.candidate)
    self.assertRaises(TypeError, meteor_score, self.reference, str_candidate)