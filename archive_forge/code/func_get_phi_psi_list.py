import warnings
from Bio.Data.PDBData import nucleic_letters_3to1
from Bio.Data.PDBData import nucleic_letters_3to1_extended
from Bio.Data.PDBData import protein_letters_3to1
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.vectors import calc_dihedral, calc_angle
from Bio.Seq import Seq
def get_phi_psi_list(self):
    """Return the list of phi/psi dihedral angles."""
    ppl = []
    lng = len(self)
    for i in range(lng):
        res = self[i]
        try:
            n = res['N'].get_vector()
            ca = res['CA'].get_vector()
            c = res['C'].get_vector()
        except Exception:
            ppl.append((None, None))
            res.xtra['PHI'] = None
            res.xtra['PSI'] = None
            continue
        if i > 0:
            rp = self[i - 1]
            try:
                cp = rp['C'].get_vector()
                phi = calc_dihedral(cp, n, ca, c)
            except Exception:
                phi = None
        else:
            phi = None
        if i < lng - 1:
            rn = self[i + 1]
            try:
                nn = rn['N'].get_vector()
                psi = calc_dihedral(n, ca, c, nn)
            except Exception:
                psi = None
        else:
            psi = None
        ppl.append((phi, psi))
        res.xtra['PHI'] = phi
        res.xtra['PSI'] = psi
    return ppl