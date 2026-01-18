import numpy as np
from ase.io.fortranfile import FortranFile
def read_vca_synth_block(filename, species_number=None):
    """ Read the SyntheticAtoms block from the output of the
    'fractional' siesta utility.

    Parameters:
        - filename: String with '.synth' output from fractional.
        - species_number: Optional argument to replace override the
                          species number in the text block.

    Returns: A string that can be inserted into the main '.fdf-file'.
    """
    with open(filename, 'r') as fd:
        lines = fd.readlines()
    lines = lines[1:-1]
    if species_number is not None:
        lines[0] = '%d\n' % species_number
    block = ''.join(lines).strip()
    return block