import itertools
from collections import Counter, defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdqueries
from . import utils
def reassignReactionRoles(smi):
    rxn = AllChem.ReactionFromSmarts(smi, useSmiles=True)
    new_rxn = reassignRXNRoles(rxn)
    if new_rxn is None:
        return ''
    smi_new = AllChem.ReactionToSmiles(new_rxn)
    return smi_new