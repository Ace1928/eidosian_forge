import re
from collections import Counter, defaultdict, namedtuple
import numpy as np
import seaborn as sns
from numpy import linalg
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
def generateChiralDescriptorsForAllCenters(mol, verbose=False):
    """
    Generates descriptors for all chiral centers in the molecule.
    Details of these descriptors are described in: 
    Schneider et al., Chiral Cliffs: Investigating the Influence of Chirality on Binding Affinity
    https://doi.org/10.1002/cmdc.201700798. 
    >>> # test molecules are taken from the publication above (see Figure 3 and Figure 8)
    >>> testmols = {
    ...   "CHEMBL319180" : 'CCCN1C(=O)[C@@H](NC(=O)Nc2cccc(C)c2)N=C(N3CCN(C)CC3)c4ccccc14',
    ...   }
    >>> mol = Chem.MolFromSmiles(testmols['CHEMBL319180'])
    >>> desc = generateChiralDescriptorsForAllCenters(mol)
    >>> desc.keys()
    dict_keys([6])
    >>> desc[6]['arLevel2']
    0
    >>> desc[6]['s4_pathLength']
    7
    >>> desc[6]['maxDist']
    14
    >>> desc[6]['maxDistfromCC']
    7
    """
    desc = {}
    dists = Chem.GetDistanceMatrix(mol)
    for idxChiral, _ in Chem.FindMolChiralCenters(mol):
        desc[idxChiral] = calculateChiralDescriptors(mol, idxChiral, dists, verbose=False)
    return desc