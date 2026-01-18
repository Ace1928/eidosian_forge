import io
import re
import pathlib
import numpy as np
from ase.calculators.lammps import convert
def extract_atom_quantities(raw_datafile_contents):
    atoms_block = extract_section(raw_datafile_contents, 'Atoms')
    charges = []
    positions = []
    travels = []
    RE_ATOM_LINE = re.compile('\\s*[0-9]+\\s+[0-9]+\\s+[0-9]+\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s?([0-9-]+)?\\s?([0-9-]+)?\\s?([0-9-]+)?')
    for atom_line in atoms_block.splitlines():
        q, x, y, z, *travel = RE_ATOM_LINE.match(atom_line).groups()
        charges.append(float(q))
        positions.append(list(map(float, [x, y, z])))
        if None not in travel:
            travels.append(list(map(int, travel)))
        else:
            travels.append(None)
    return (charges, positions, travels)