import sys
import weakref
from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions
def getUltimateParents(self):
    """ returns all the nodes in the hierarchy tree that contain this
            node as a child
        """
    if not self.parents:
        res = [self]
    else:
        res = []
        for p in self.parents.values():
            for uP in p.getUltimateParents():
                if uP not in res:
                    res.append(uP)
    return res