import math
from collections import namedtuple
from rdkit import Chem
from rdkit.Chem import Crippen, MolSurf
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem.ChemUtils.DescriptorUtilities import setDescriptorVersion
@setDescriptorVersion(version='1.1.0')
def qed(mol, w=WEIGHT_MEAN, qedProperties=None):
    """ Calculate the weighted sum of ADS mapped properties

  some examples from the QED paper, reference values from Peter G's original implementation
  >>> m = Chem.MolFromSmiles('N=C(CCSCc1csc(N=C(N)N)n1)NS(N)(=O)=O')
  >>> qed(m)
  0.253...
  >>> m = Chem.MolFromSmiles('CNC(=NCCSCc1nc[nH]c1C)NC#N')
  >>> qed(m)
  0.234...
  >>> m = Chem.MolFromSmiles('CCCCCNC(=N)NN=Cc1c[nH]c2ccc(CO)cc12')
  >>> qed(m)
  0.234...
  """
    if qedProperties is None:
        qedProperties = properties(mol)
    d = [ads(pi, adsParameters[name]) for name, pi in qedProperties._asdict().items()]
    t = sum((wi * math.log(di) for wi, di in zip(w, d)))
    return math.exp(t / sum(w))