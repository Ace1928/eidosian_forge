import warnings
from Bio.Data.PDBData import nucleic_letters_3to1
from Bio.Data.PDBData import nucleic_letters_3to1_extended
from Bio.Data.PDBData import protein_letters_3to1
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.vectors import calc_dihedral, calc_angle
from Bio.Seq import Seq
def get_tau_list(self):
    """List of tau torsions angles for all 4 consecutive Calpha atoms."""
    ca_list = self.get_ca_list()
    tau_list = []
    for i in range(len(ca_list) - 3):
        atom_list = (ca_list[i], ca_list[i + 1], ca_list[i + 2], ca_list[i + 3])
        v1, v2, v3, v4 = (a.get_vector() for a in atom_list)
        tau = calc_dihedral(v1, v2, v3, v4)
        tau_list.append(tau)
        res = ca_list[i + 2].get_parent()
        res.xtra['TAU'] = tau
    return tau_list