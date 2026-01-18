import sys
import weakref
from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions
def testSFNetIssue1804418(self):
    m = Chem.MolFromSmiles('C1CCCCN1CCCC')
    res = RecapDecompose(m)
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 2)
    ks = res.GetLeaves().keys()
    self.assertTrue('*N1CCCCC1' in ks)
    self.assertTrue('*CCCC' in ks)