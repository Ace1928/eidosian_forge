import tempfile
import shutil
import os
import numpy as np
from numpy import pi
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.odr import (Data, Model, ODR, RealData, OdrStop, OdrWarning,
def test_implicit(self):
    implicit_mod = Model(self.implicit_fcn, implicit=1, meta=dict(name='Sample Implicit Model', ref='ODRPACK UG, pg. 49'))
    implicit_dat = Data([[0.5, 1.2, 1.6, 1.86, 2.12, 2.36, 2.44, 2.36, 2.06, 1.74, 1.34, 0.9, -0.28, -0.78, -1.36, -1.9, -2.5, -2.88, -3.18, -3.44], [-0.12, -0.6, -1.0, -1.4, -2.54, -3.36, -4.0, -4.75, -5.25, -5.64, -5.97, -6.32, -6.44, -6.44, -6.41, -6.25, -5.88, -5.5, -5.24, -4.86]], 1)
    implicit_odr = ODR(implicit_dat, implicit_mod, beta0=[-1.0, -3.0, 0.09, 0.02, 0.08])
    out = implicit_odr.run()
    assert_array_almost_equal(out.beta, np.array([-0.9993809167281279, -2.9310484652026476, 0.0875730502693354, 0.0162299708984738, 0.0797537982976416]))
    assert_array_almost_equal(out.sd_beta, np.array([0.1113840353364371, 0.1097673310686467, 0.0041060738314314, 0.0027500347539902, 0.0034962501532468]))
    assert_allclose(out.cov_beta, np.array([[2.1089274602333052, -1.943768641197904, 0.07026355086834445, -0.04717526737347486, 0.052515575927380355], [-1.943768641197904, 2.0481509222414456, -0.06160051585305731, 0.04626882780623293, -0.05882230750139147], [0.07026355086834445, -0.06160051585305731, 0.002865954256157931, -0.001462866226001449, 0.0014528860663055824], [-0.04717526737347486, 0.04626882780623293, -0.001462866226001449, 0.0012855592885514335, -0.0012692942951415293], [0.052515575927380355, -0.05882230750139147, 0.0014528860663055824, -0.0012692942951415293, 0.0020778813389755596]]), rtol=1e-06, atol=2e-06)