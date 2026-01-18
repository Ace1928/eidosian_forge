import re
import numpy as np
from ase.units import Bohr, Angstrom, Hartree, eV, Debye
def read_static_info_kpoints(fd):
    for line in fd:
        if line.startswith('List of k-points'):
            break
    tokens = next(fd).split()
    assert tokens == ['ik', 'k_x', 'k_y', 'k_z', 'Weight']
    bar = next(fd)
    assert bar.startswith('---')
    kpts = []
    weights = []
    for line in fd:
        m = re.match('\\s*\\d+\\s*(\\S+)\\s*(\\S+)\\s*(\\S+)\\s*(\\S+)', line)
        if m is None:
            break
        kxyz = m.group(1, 2, 3)
        weight = m.group(4)
        kpts.append(kxyz)
        weights.append(weight)
    ibz_k_points = np.array(kpts, float)
    k_point_weights = np.array(weights, float)
    return dict(ibz_k_points=ibz_k_points, k_point_weights=k_point_weights)