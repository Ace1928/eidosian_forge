import pytest
import math
import rpy2.rinterface as ri
def test_R_to_NAComplex():
    r_na_complex = ri.evalr('NA_complex_')[0]
    assert math.isnan(r_na_complex.real)
    assert math.isnan(r_na_complex.imag)