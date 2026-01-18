import sys
import weakref
from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions
def testAromCAromCRxn(self):
    m = Chem.MolFromSmiles('c1ccccc1c1ncccc1')
    res = RecapDecompose(m)
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 2)
    ks = res.GetLeaves().keys()
    self.assertTrue('*c1ccccc1' in ks)
    self.assertTrue('*c1ccccn1' in ks)
    m = Chem.MolFromSmiles('c1ccccc1C1CC1')
    res = RecapDecompose(m)
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 0)