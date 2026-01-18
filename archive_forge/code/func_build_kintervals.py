import numpy as np
from collections import namedtuple
from ase.geometry.dimensionality import rank_determination
from ase.geometry.dimensionality import topology_scaling
from ase.geometry.dimensionality.bond_generator import next_bond
def build_kintervals(atoms, method_name):
    """The interval analysis is performed by inserting bonds one at a time
    until the component analysis finds a single component."""
    method = {'RDA': rank_determination.RDA, 'TSA': topology_scaling.TSA}[method_name]
    assert all([e in [0, 1] for e in atoms.pbc])
    num_atoms = len(atoms)
    intervals = []
    kprev = 0
    calc = method(num_atoms)
    hprev = calc.check()
    components_prev, cdim_prev = calc.get_components()
    '\n    The end state is a single component, whose dimensionality depends on\n    the periodic boundary conditions:\n    '
    end_state = np.zeros(4)
    end_dim = sum(atoms.pbc)
    end_state[end_dim] = 1
    end_state = tuple(end_state)
    for k, i, j, offset in next_bond(atoms):
        calc.insert_bond(i, j, offset)
        h = calc.check()
        if h == hprev:
            continue
        components, cdim = calc.get_components()
        if k != kprev:
            intervals.append(build_kinterval(kprev, k, hprev, components_prev, cdim_prev))
        kprev = k
        hprev = h
        components_prev = components
        cdim_prev = cdim
        if h == end_state:
            intervals.append(build_kinterval(k, float('inf'), h, components, cdim))
            return intervals