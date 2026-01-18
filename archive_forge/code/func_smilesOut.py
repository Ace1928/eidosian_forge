from rdkit import Chem
from rdkit.VLib.Output import OutputNode as BaseOutputNode
def smilesOut(self, mol):
    self._nDumped += 1
    if isinstance(mol, (tuple, list)):
        args = mol
        mol = args[0]
        if len(args) > 1:
            args = list(args[1:])
        else:
            args = []
    else:
        args = []
    if self._idField and mol.HasProp(self._idField):
        label = mol.GetProp(self._idField)
    else:
        label = str(self._nDumped)
    smi = Chem.MolToSmiles(mol)
    outp = [label, smi] + args
    return '%s\n' % self._delim.join(outp)