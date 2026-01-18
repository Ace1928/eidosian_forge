from copy import copy
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.SCOP.Residues import Residues
def getAtoms(self, pdb_handle, out_handle):
    """Extract all relevant ATOM and HETATOM records from a PDB file.

        The PDB file is scanned for ATOM and HETATOM records. If the
        chain ID, residue ID (seqNum and iCode), and residue type match
        a residue in this sequence map, then the record is echoed to the
        output handle.

        This is typically used to find the coordinates of a domain, or other
        residue subset.

        Arguments:
         - pdb_handle -- A handle to the relevant PDB file.
         - out_handle -- All output is written to this file like object.

        """
    resSet = {}
    for r in self.res:
        if r.atom == 'X':
            continue
        chainid = r.chainid
        if chainid == '_':
            chainid = ' '
        resid = r.resid
        resSet[chainid, resid] = r
    resFound = {}
    for line in pdb_handle:
        if line.startswith(('ATOM  ', 'HETATM')):
            chainid = line[21:22]
            resid = line[22:27].strip()
            key = (chainid, resid)
            if key in resSet:
                res = resSet[key]
                atom_aa = res.atom
                resName = line[17:20]
                if resName in protein_letters_3to1_extended:
                    if protein_letters_3to1_extended[resName] == atom_aa:
                        out_handle.write(line)
                        resFound[key] = res
    if len(resSet) != len(resFound):
        raise RuntimeError('Could not find at least one ATOM or HETATM record for each and every residue in this sequence map.')