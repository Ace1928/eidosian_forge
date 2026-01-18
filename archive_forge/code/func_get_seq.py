import subprocess
import os
from Bio.PDB.Polypeptide import is_aa
def get_seq(self):
    """Return secondary structure string."""
    return self.ss_seq