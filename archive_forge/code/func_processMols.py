import math
import os.path as op
import pickle
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
def processMols(mols):
    print('smiles\tName\tsa_score')
    for i, m in enumerate(mols):
        if m is None:
            continue
        s = calculateScore(m)
        smiles = Chem.MolToSmiles(m)
        print(smiles + '\t' + m.GetProp('_Name') + '\t%3f' % s)