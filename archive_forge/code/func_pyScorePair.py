from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.AtomPairs import Utils
from rdkit.Chem.rdMolDescriptors import (GetAtomPairFingerprint,
def pyScorePair(at1, at2, dist, atomCodes=None, includeChirality=False):
    """ Returns a score for an individual atom pair.

  >>> from rdkit import Chem
  >>> m = Chem.MolFromSmiles('CCCCC')
  >>> c1 = Utils.GetAtomCode(m.GetAtomWithIdx(0))
  >>> c2 = Utils.GetAtomCode(m.GetAtomWithIdx(1))
  >>> c3 = Utils.GetAtomCode(m.GetAtomWithIdx(2))
  >>> t = 1 | min(c1,c2) << numPathBits | max(c1,c2) << (rdMolDescriptors.AtomPairsParameters.codeSize + numPathBits)
  >>> pyScorePair(m.GetAtomWithIdx(0), m.GetAtomWithIdx(1), 1)==t
  1
  >>> pyScorePair(m.GetAtomWithIdx(1), m.GetAtomWithIdx(0), 1)==t
  1
  >>> t = 2 | min(c1,c3) << numPathBits | max(c1,c3) << (rdMolDescriptors.AtomPairsParameters.codeSize + numPathBits)
  >>> pyScorePair(m.GetAtomWithIdx(0),m.GetAtomWithIdx(2),2)==t
  1
  >>> pyScorePair(m.GetAtomWithIdx(0),m.GetAtomWithIdx(2),2,
  ...  atomCodes=(Utils.GetAtomCode(m.GetAtomWithIdx(0)), Utils.GetAtomCode(m.GetAtomWithIdx(2)))) == t
  1

  """
    if not atomCodes:
        code1 = Utils.GetAtomCode(at1, includeChirality=includeChirality)
        code2 = Utils.GetAtomCode(at2, includeChirality=includeChirality)
    else:
        code1, code2 = atomCodes
    codeSize = rdMolDescriptors.AtomPairsParameters.codeSize
    if includeChirality:
        codeSize += rdMolDescriptors.AtomPairsParameters.numChiralBits
    accum = int(dist) % _maxPathLen
    accum |= min(code1, code2) << numPathBits
    accum |= max(code1, code2) << codeSize + numPathBits
    return accum