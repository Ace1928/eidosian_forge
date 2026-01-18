import io
import re
import pathlib
import numpy as np
from ase.calculators.lammps import convert
def extract_velocities(raw_datafile_contents):
    """
    NOTE: Assumes metal units are used in data file
    """
    velocities_block = extract_section(raw_datafile_contents, 'Velocities')
    RE_VELOCITY = re.compile('\\s*[0-9]+\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)')
    velocities = []
    for velocities_line in velocities_block.splitlines():
        v = RE_VELOCITY.match(velocities_line).groups()
        velocities.append(list(map(float, v)))
    velocities = convert(np.asarray(velocities), 'velocity', 'metal', 'ASE')
    return velocities