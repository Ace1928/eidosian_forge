from math import pi, sqrt
import numpy as np
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.parallel import world
from ase.utils.cext import cextension
@cextension
def lti_dos(simplices, eigs, weights, energies, dos, world):
    shape = eigs.shape[:3]
    nweights = weights.shape[-1]
    dos[:] = 0.0
    n = -1
    for index in np.indices(shape).reshape((3, -1)).T:
        n += 1
        if n % world.size != world.rank:
            continue
        i = ((index + simplices) % shape).T
        E = eigs[i[0], i[1], i[2]].reshape((4, -1))
        W = weights[i[0], i[1], i[2]].reshape((4, -1, nweights))
        for e, w in zip(E.T, W.transpose((1, 0, 2))):
            lti_dos1(e, w, energies, dos)
    dos /= 6.0
    world.sum(dos)