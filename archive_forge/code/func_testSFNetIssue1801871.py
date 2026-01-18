import sys
import weakref
from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions
def testSFNetIssue1801871(self):
    m = Chem.MolFromSmiles('c1ccccc1OC(Oc1ccccc1)Oc1ccccc1')
    res = RecapDecompose(m)
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 2)
    ks = res.GetLeaves().keys()
    self.assertFalse('*C(*)*' in ks)
    self.assertTrue('*c1ccccc1' in ks)
    self.assertTrue('*C(*)Oc1ccccc1' in ks)