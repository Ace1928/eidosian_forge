import copy
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
def getNumNegativeCounts(fp):
    count = 0
    for k, v in fp.GetNonzeroElements().items():
        if v < 0:
            count += abs(v)
    return count