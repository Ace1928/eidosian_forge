import re
import sys
from optparse import OptionParser
from rdkit import Chem
from rdkit.Chem import AllChem
def smiles_to_smarts(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        sys.stderr.write("Can't generate mol for: %s\n" % smi)
        return None
    for atom in mol.GetAtoms():
        atom.SetIsotope(42)
    smarts = Chem.MolToSmiles(mol, isomericSmiles=True)
    smarts = re.sub('\\[42', '[', smarts)
    return smarts