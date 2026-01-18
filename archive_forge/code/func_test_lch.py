import unittest
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wnic
def test_lch(self):
    self.assertEqual(S('person.n.01').lowest_common_hypernyms(S('dog.n.01')), [S('organism.n.01')])
    self.assertEqual(S('woman.n.01').lowest_common_hypernyms(S('girlfriend.n.02')), [S('woman.n.01')])