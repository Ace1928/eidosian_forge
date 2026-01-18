import unittest
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wnic
def test_misc_relations(self):
    self.assertEqual(S('snore.v.1').entailments(), [S('sleep.v.01')])
    self.assertEqual(S('heavy.a.1').similar_tos(), [S('dense.s.03'), S('doughy.s.01'), S('heavier-than-air.s.01'), S('hefty.s.02'), S('massive.s.04'), S('non-buoyant.s.01'), S('ponderous.s.02')])
    self.assertEqual(S('light.a.1').attributes(), [S('weight.n.01')])
    self.assertEqual(S('heavy.a.1').attributes(), [S('weight.n.01')])
    self.assertEqual(L('English.a.1.English').pertainyms(), [L('england.n.01.England')])