import subprocess
import os
from Bio.PDB.Polypeptide import is_aa
def psea(pname):
    """Parse PSEA output file."""
    fname = run_psea(pname)
    start = 0
    ss = ''
    with open(fname) as fp:
        for line in fp:
            if line[0:6] == '>p-sea':
                start = 1
                continue
            if not start:
                continue
            if line[0] == '\n':
                break
            ss = ss + line[0:-1]
    return ss