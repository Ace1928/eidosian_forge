import itertools
import numpy as np
from ase.lattice import bravais_lattices, UnconventionalLattice, bravais_names
from ase.cell import Cell
def find_all_niggli_ops(length_grid, angle_grid, lattices=None):
    all_niggli_ops = {}
    if lattices is None:
        lattices = [name for name in bravais_names if name not in ['MCL', 'MCLC', 'TRI']]
    for latname in lattices:
        latcls = bravais_lattices[latname]
        if latcls.ndim < 3:
            continue
        print('Working on {}...'.format(latname))
        niggli_ops = find_niggli_ops(latcls, length_grid, angle_grid)
        print('Found {} ops for {}'.format(len(niggli_ops), latname))
        for key, count in niggli_ops.items():
            print('  {:>40}: {}'.format(str(np.array(key)), count))
        print()
        all_niggli_ops[latname] = niggli_ops
    return all_niggli_ops