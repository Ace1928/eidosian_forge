import unittest
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wnic
def test_derivationally_related_forms(self):
    self.assertEqual(L('zap.v.03.nuke').derivationally_related_forms(), [L('atomic_warhead.n.01.nuke')])
    self.assertEqual(L('zap.v.03.atomize').derivationally_related_forms(), [L('atomization.n.02.atomization')])
    self.assertEqual(L('zap.v.03.atomise').derivationally_related_forms(), [L('atomization.n.02.atomisation')])
    self.assertEqual(L('zap.v.03.zap').derivationally_related_forms(), [])