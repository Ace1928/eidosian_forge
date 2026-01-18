from Bio.PDB.Entity import Entity
def get_residues(self):
    """Return residues from chains."""
    for c in self.get_chains():
        yield from c