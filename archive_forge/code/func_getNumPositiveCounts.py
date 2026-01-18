import copy
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
def getNumPositiveCounts(fp):
    count = 0
    for k, v in fp.GetNonzeroElements().items():
        if v > 0:
            count += v
    return count