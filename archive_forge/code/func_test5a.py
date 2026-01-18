import copy
import re
import sys
from rdkit import Chem
from rdkit import RDRandom as random
from rdkit.Chem import rdChemReactions as Reactions
def test5a(self):
    allNodes = set()
    frags = ['[3*]O[3*]', '[16*]c1ccccc1']
    frags = [Chem.MolFromSmiles(x) for x in frags]
    res = BRICSBuild(frags)
    self.assertTrue(res)
    res = list(res)
    smis = [Chem.MolToSmiles(x, True) for x in res]
    self.assertTrue(len(smis) == 2, smis)
    self.assertTrue('c1ccc(Oc2ccccc2)cc1' in smis)
    self.assertTrue('c1ccc(-c2ccccc2)cc1' in smis)