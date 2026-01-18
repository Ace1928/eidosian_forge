import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.Polypeptide import PPBuilder
def add_residue(self, resname, ca_coord):
    """Add a residue.

        :param resname: residue name (eg. GLY).
        :type resname: string

        :param ca_coord: the c-alpha coordinates of the residues
        :type ca_coord: NumPy array with length 3
        """
    if self.counter >= self.length:
        raise PDBException('Fragment boundary exceeded.')
    self.resname_list.append(resname)
    self.coords_ca[self.counter] = ca_coord
    self.counter = self.counter + 1