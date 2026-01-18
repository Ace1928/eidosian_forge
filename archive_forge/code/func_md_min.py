from time import time
from math import sqrt, pi
import numpy as np
from ase.parallel import paropen
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.transport.tools import dagger, normalize
from ase.io.jsonio import read_json, write_json
def md_min(func, step=0.25, tolerance=1e-06, verbose=False, **kwargs):
    if verbose:
        print('Localize with step =', step, 'and tolerance =', tolerance)
        t = -time()
    fvalueold = 0.0
    fvalue = fvalueold + 10
    count = 0
    V = np.zeros(func.get_gradients().shape, dtype=complex)
    while abs((fvalue - fvalueold) / fvalue) > tolerance:
        fvalueold = fvalue
        dF = func.get_gradients()
        V *= (dF * V.conj()).real > 0
        V += step * dF
        func.step(V, **kwargs)
        fvalue = func.get_functional_value()
        if fvalue < fvalueold:
            step *= 0.5
        count += 1
        if verbose:
            print('MDmin: iter=%s, step=%s, value=%s' % (count, step, fvalue))
    if verbose:
        t += time()
        print('%d iterations in %0.2f seconds (%0.2f ms/iter), endstep = %s' % (count, t, t * 1000.0 / count, step))