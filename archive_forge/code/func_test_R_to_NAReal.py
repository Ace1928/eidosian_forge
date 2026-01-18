import pytest
import math
import rpy2.rinterface as ri
def test_R_to_NAReal():
    r_na_real = ri.evalr('NA_real_')[0]
    assert math.isnan(r_na_real)