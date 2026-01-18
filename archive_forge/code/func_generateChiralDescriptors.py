import re
from collections import Counter, defaultdict, namedtuple
import numpy as np
import seaborn as sns
from numpy import linalg
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
def generateChiralDescriptors(mol, verbose=False):
    """
    Generates descriptors for the 'first' chiral centers in the molecule.
    Details of these descriptors are described in: 
    Schneider et al., Chiral Cliffs: Investigating the Influence of Chirality on Binding Affinity
    https://doi.org/10.1002/cmdc.201700798. 
    >>> # test molecules are taken from the publication above (see Figure 3 and Figure 8)
    >>> testmols = {
    ...   "CHEMBL319180" : 'CCCN1C(=O)[C@@H](NC(=O)Nc2cccc(C)c2)N=C(N3CCN(C)CC3)c4ccccc14',
    ...   "CHEMBL3350250" : 'CC(C)[C@@]1(CCc2ccc(O)cc2)CC(=O)C(=C(O)O1)Sc3cc(C)c(NS(=O)(=O)c4ccc(cn4)C(F)(F)F)cc3C(C)(C)C',
    ...   "CHEMBL3698720" : 'N[C@@H]1CCN(C1)c2cnc(Nc3ncc4c5ccncc5n(C6CCCC6)c4n3)cn2'
    ...   }
    >>> mol = Chem.MolFromSmiles(testmols['CHEMBL319180'])
    >>> desc = generateChiralDescriptors(mol)
    >>> desc['arLevel2']
    0
    >>> desc['s4_pathLength']
    7
    >>> desc['maxDist']
    14
    >>> desc['maxDistfromCC']
    7
    >>> mol = Chem.MolFromSmiles(testmols['CHEMBL3350250'])
    >>> desc = generateChiralDescriptors(mol)
    >>> desc['nLevel8']
    5
    >>> desc['s1_pathLength']
    2
    >>> desc['s2_size']
    9.0
    >>> desc['numRotBonds']
    9
    >>> mol = Chem.MolFromSmiles(testmols['CHEMBL3698720'])
    >>> desc = generateChiralDescriptors(mol)
    >>> desc['s3_numRotBonds']
    0
    >>> desc['s34_size']
    29.0
    >>> desc['phLevel2']
    1
    >>> desc['s2_numSharedNeighbors']
    0
    
    """
    dists = Chem.GetDistanceMatrix(mol)
    idxChiral = Chem.FindMolChiralCenters(mol)[0][0]
    return calculateChiralDescriptors(mol, idxChiral, dists, verbose=False)