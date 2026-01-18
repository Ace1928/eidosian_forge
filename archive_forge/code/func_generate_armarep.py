import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy import array
def generate_armarep(filen='testsave.py'):
    from mlabwrap import mlab
    res_armarep = HoldIt('armarep')
    res_armarep.ar = np.array([1.0, -0.5, +0.8])
    res_armarep.ma = np.array([1.0, -0.6, 0.08])
    res_armarep.marep = mlab.garchma(-res_armarep.ar[1:], res_armarep.ma[1:], 20)
    res_armarep.arrep = mlab.garchar(-res_armarep.ar[1:], res_armarep.ma[1:], 20)
    res_armarep.save(filename=filen, header=False, comment="''mlab.garchma(-res_armarep.ar[1:], res_armarep.ma[1:], 20)\n" + "mlab.garchar(-res_armarep.ar[1:], res_armarep.ma[1:], 20)''")