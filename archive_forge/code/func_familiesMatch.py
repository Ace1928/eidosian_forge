import copy
from rdkit.Chem.FeatMaps import FeatMaps
def familiesMatch(f1, f2):
    return f1.GetFamily() == f2.GetFamily()