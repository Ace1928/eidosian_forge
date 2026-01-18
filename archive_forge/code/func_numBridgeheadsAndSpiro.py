import math
import os.path as op
import pickle
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return (nBridgehead, nSpiro)