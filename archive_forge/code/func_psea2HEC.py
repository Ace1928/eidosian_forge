import subprocess
import os
from Bio.PDB.Polypeptide import is_aa
def psea2HEC(pseq):
    """Translate PSEA secondary structure string into HEC."""
    seq = []
    for ss in pseq:
        if ss == 'a':
            n = 'H'
        elif ss == 'b':
            n = 'E'
        elif ss == 'c':
            n = 'C'
        seq.append(n)
    return seq