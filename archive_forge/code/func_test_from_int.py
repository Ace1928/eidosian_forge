import subprocess
import pytest
import sys
import os
import textwrap
import rpy2.rinterface as ri
def test_from_int():
    sexp = ri.vector([1], ri.RTYPES.INTSXP)
    isInteger = ri.globalenv.find('is.integer')
    assert isInteger(sexp)[0]
    with pytest.raises(ValueError):
        ri.vector(['a'], ri.RTYPES.INTSXP)