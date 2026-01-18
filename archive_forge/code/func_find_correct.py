import re
import sys
from rdkit import Chem
from rdkit.Chem import rdMMPA
def find_correct(f_array):
    core = ''
    side_chains = ''
    for f in f_array:
        attachments = f.count('*')
        if attachments == 1:
            side_chains = '%s.%s' % (side_chains, f)
        else:
            core = f
    side_chains = side_chains.lstrip('.')
    temp = Chem.MolFromSmiles(side_chains)
    side_chains = Chem.MolToSmiles(temp, isomericSmiles=True)
    temp = Chem.MolFromSmiles(core)
    core = Chem.MolToSmiles(temp, isomericSmiles=True)
    return (core, side_chains)