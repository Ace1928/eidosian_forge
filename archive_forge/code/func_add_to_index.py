import re
import sys
from optparse import OptionParser
from rdkit import Chem
def add_to_index(smi, attachments, cmpd_heavy):
    result = False
    core_size = heavy_atom_count(smi) - attachments
    if use_ratio:
        core_ratio = float(core_size) / float(cmpd_heavy)
        if core_ratio <= ratio:
            result = True
    elif core_size <= max_size:
        result = True
    return result