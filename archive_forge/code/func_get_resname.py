from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.Entity import Entity, DisorderedEntityWrapper
def get_resname(self):
    """Return the residue name."""
    return self.resname