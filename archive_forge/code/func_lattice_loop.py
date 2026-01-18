import itertools
import numpy as np
from ase.lattice import bravais_lattices, UnconventionalLattice, bravais_names
from ase.cell import Cell
def lattice_loop(latcls, length_grid, angle_grid):
    """Yield all lattices defined by the length and angle grids."""
    param_grids = []
    for varname in latcls.parameters:
        if latcls.name in ['MCL', 'MCLC']:
            special_var = 'c'
        else:
            special_var = 'a'
        if varname == special_var:
            values = np.ones(1)
        elif varname in 'abc':
            values = length_grid
        elif varname in ['alpha', 'beta', 'gamma']:
            values = angle_grid
        else:
            raise ValueError(varname)
        param_grids.append(values)
    for latpars in itertools.product(*param_grids):
        kwargs = dict(zip(latcls.parameters, latpars))
        try:
            lat = latcls(**kwargs)
        except (UnconventionalLattice, AssertionError):
            pass
        else:
            yield lat