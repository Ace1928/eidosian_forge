import copy
import re
import sys
from rdkit import Chem
from rdkit import RDRandom as random
from rdkit.Chem import rdChemReactions as Reactions
def test12(self):
    m = Chem.MolFromSmiles('CCS(=O)(=O)NCC')
    res = list(FindBRICSBonds(m))
    self.assertEqual(len(res), 2, res)
    atIds = [x[0] for x in res]
    atIds.sort()
    self.assertEqual(atIds, [(5, 2), (6, 5)])