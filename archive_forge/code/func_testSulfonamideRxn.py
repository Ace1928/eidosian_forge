import sys
import weakref
from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions
def testSulfonamideRxn(self):
    m = Chem.MolFromSmiles('CCCNS(=O)(=O)CC')
    res = RecapDecompose(m)
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 2)
    ks = res.GetLeaves().keys()
    self.assertTrue('*NCCC' in ks)
    self.assertTrue('*S(=O)(=O)CC' in ks)
    m = Chem.MolFromSmiles('c1cccn1S(=O)(=O)CC')
    res = RecapDecompose(m)
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 2)
    ks = res.GetLeaves().keys()
    self.assertTrue('*n1cccc1' in ks)
    self.assertTrue('*S(=O)(=O)CC' in ks)
    m = Chem.MolFromSmiles('C1CNS(=O)(=O)CC1')
    res = RecapDecompose(m)
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 0)