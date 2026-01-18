from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.Entity import Entity, DisorderedEntityWrapper
def get_unpacked_list(self):
    """Return the list of all atoms, unpack DisorderedAtoms."""
    atom_list = self.get_list()
    undisordered_atom_list = []
    for atom in atom_list:
        if atom.is_disordered():
            undisordered_atom_list += atom.disordered_get_list()
        else:
            undisordered_atom_list.append(atom)
    return undisordered_atom_list