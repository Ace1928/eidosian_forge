import pytest
import math
import rpy2.rinterface as ri
def test_R_to_NALogical():
    r_na_lgl = ri.evalr('NA')[0]
    assert r_na_lgl is ri.NA