import os
import tempfile
import shutil
import subprocess
import warnings
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.AbstractPropertyMap import (
def process_asa_data(rsa_data):
    """Process the .asa output file: atomic level SASA data."""
    naccess_atom_dict = {}
    for line in rsa_data:
        full_atom_id = line[12:16]
        atom_id = full_atom_id.strip()
        chainid = line[21]
        resseq = int(line[22:26])
        icode = line[26]
        res_id = (' ', resseq, icode)
        id = (chainid, res_id, atom_id)
        asa = line[54:62]
        naccess_atom_dict[id] = asa
    return naccess_atom_dict