import tempfile
import shutil
import os
import numpy as np
from numpy import pi
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.odr import (Data, Model, ODR, RealData, OdrStop, OdrWarning,
def test_explicit(self):
    explicit_mod = Model(self.explicit_fcn, fjacb=self.explicit_fjb, fjacd=self.explicit_fjd, meta=dict(name='Sample Explicit Model', ref='ODRPACK UG, pg. 39'))
    explicit_dat = Data([0.0, 0.0, 5.0, 7.0, 7.5, 10.0, 16.0, 26.0, 30.0, 34.0, 34.5, 100.0], [1265.0, 1263.6, 1258.0, 1254.0, 1253.0, 1249.8, 1237.0, 1218.0, 1220.6, 1213.8, 1215.5, 1212.0])
    explicit_odr = ODR(explicit_dat, explicit_mod, beta0=[1500.0, -50.0, -0.1], ifixx=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    explicit_odr.set_job(deriv=2)
    explicit_odr.set_iprint(init=0, iter=0, final=0)
    out = explicit_odr.run()
    assert_array_almost_equal(out.beta, np.array([1264.6548050648876, -54.018409956678255, -0.08784971216525372]))
    assert_array_almost_equal(out.sd_beta, np.array([1.0349270280543437, 1.583997785262061, 0.0063321988657267]))
    assert_array_almost_equal(out.cov_beta, np.array([[0.4494959237900304, -0.3742197689036474, -0.0008097821746846891], [-0.3742197689036474, 1.0529686462751804, -0.0019453521827942002], [-0.0008097821746846891, -0.0019453521827942002, 1.6827336938454476e-05]]))