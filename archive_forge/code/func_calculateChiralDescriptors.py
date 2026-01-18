import re
from collections import Counter, defaultdict, namedtuple
import numpy as np
import seaborn as sns
from numpy import linalg
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
def calculateChiralDescriptors(mol, idxChiral, dists, verbose=False):
    desc = {}
    subs, sharedNeighbors, maxShell = determineAtomSubstituents(idxChiral, mol, dists, verbose)
    sizes = calcSizeSubstituents(mol, subs, sharedNeighbors, maxShell)
    paths = dists[idxChiral]
    desc['numAtoms'] = mol.GetNumAtoms()
    desc['numBonds'] = mol.GetNumBonds()
    desc['numRotBonds'] = AllChem.CalcNumRotatableBonds(mol)
    desc['ringChiralCenter'] = int(mol.GetAtomWithIdx(idxChiral).IsInRing())
    desc['meanDist'] = np.sum(dists) / ((desc['numAtoms'] - 1) * desc['numAtoms'])
    desc['maxDist'] = int(np.max(dists))
    desc['meanDistFromCC'] = np.sum(paths) / (desc['numAtoms'] - 1)
    desc['maxDistfromCC'] = int(np.max(paths))
    nlevels = Counter(paths.astype(int))
    for i in range(1, 11):
        desc['nLevel' + str(i)] = nlevels[i]
    for i in range(1, 4):
        desc['phLevel' + str(i)] = len([n for n, j in enumerate(paths) if j == i and mol.GetAtomWithIdx(n).GetAtomicNum() in [7, 8]])
    for i in range(1, 4):
        desc['arLevel' + str(i)] = len([n for n, j in enumerate(paths) if j == i and mol.GetAtomWithIdx(n).GetIsAromatic()])
    for n, v in enumerate(sorted(sizes.values(), key=lambda x: x.size), 1):
        sn = 's' + str(n)
        desc[sn + '_size'] = v.size
        desc[sn + '_relSize'] = v.relSize
        desc[sn + '_phSize'] = v.numNO
        desc[sn + '_phRelSize'] = v.relNumNO
        desc[sn + '_phRelSize_2'] = v.relNumNO_2
        desc[sn + '_pathLength'] = v.pathLength
        desc[sn + '_relPathLength'] = v.relPathLength
        desc[sn + '_relPathLength_2'] = v.relPathLength_2
        desc[sn + '_numSharedNeighbors'] = v.sharedNeighbors
        desc[sn + '_numRotBonds'] = v.numRotBonds
        desc[sn + '_numAroBonds'] = v.numAroBonds
    desc['s34_size'] = desc['s3_size'] + desc['s4_size']
    desc['s34_phSize'] = desc['s3_phSize'] + desc['s4_phSize']
    desc['s34_relSize'] = desc['s3_relSize'] + desc['s4_relSize']
    desc['s34_phRelSize'] = desc['s3_phRelSize'] + desc['s4_phRelSize']
    desc['chiralMoment'] = calcSP3CarbonSubstituentMoment([desc['s1_size'], desc['s2_size'], desc['s3_size'], desc['s4_size']])
    desc['chiralPhMoment'] = calcSP3CarbonSubstituentMoment([desc['s1_phSize'], desc['s2_phSize'], desc['s3_phSize'], desc['s4_phSize']])
    return desc