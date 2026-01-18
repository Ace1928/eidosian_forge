import itertools
from collections import Counter, defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdqueries
from . import utils
def reassignRXNRoles(rxn):
    utils.transferAgentsToReactants(rxn)
    reacts, rAgents, pAgents = identifyReactants(rxn)
    if len(reacts) < 1:
        return None
    new_rxn = AllChem.ChemicalReaction()
    for i in range(rxn.GetNumProductTemplates()):
        new_rxn.AddProductTemplate(rxn.GetProductTemplate(i))
    for i in range(rxn.GetNumReactantTemplates()):
        if i in reacts[0]:
            new_rxn.AddReactantTemplate(rxn.GetReactantTemplate(i))
        else:
            new_rxn.AddAgentTemplate(rxn.GetReactantTemplate(i))
    return new_rxn