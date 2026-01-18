import numpy as np
from Bio.PDB.ccealign import run_cealign
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.qcprot import QCPSuperimposer
def get_guide_coord_from_structure(self, structure):
    """Return the coordinates of guide atoms in the structure.

        We use guide atoms (C-alpha and C4' atoms) since it is much faster than
        using all atoms in the calculation without a significant loss in
        accuracy.
        """
    coords = []
    for chain in sorted(structure.get_chains()):
        for resid in sorted(chain, key=_RESID_SORTER):
            if 'CA' in resid:
                coords.append(resid['CA'].coord.tolist())
            elif "C4'" in resid:
                coords.append(resid["C4'"].coord.tolist())
    if not coords:
        msg = f'Structure {structure.id} does not have any guide atoms.'
        raise PDBException(msg)
    return coords