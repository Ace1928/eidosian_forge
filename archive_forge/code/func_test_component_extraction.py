from __future__ import division
import uncertainties
import uncertainties.core as uncert_core
from uncertainties import ufloat, unumpy, test_uncertainties
from uncertainties.unumpy import core
from uncertainties.test_uncertainties import numbers_close, arrays_close
def test_component_extraction():
    """Extracting the nominal values and standard deviations from an array"""
    arr = unumpy.uarray([1, 2], [0.1, 0.2])
    assert numpy.all(unumpy.nominal_values(arr) == [1, 2])
    assert numpy.all(unumpy.std_devs(arr) == [0.1, 0.2])
    mat = unumpy.matrix(arr)
    assert numpy.all(unumpy.nominal_values(mat) == [1, 2])
    assert numpy.all(unumpy.std_devs(mat) == [0.1, 0.2])
    assert type(unumpy.nominal_values(mat)) == numpy.matrix