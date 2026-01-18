import subprocess
import pytest
import sys
import os
import textwrap
import rpy2.rinterface as ri
def test_from_complex():
    sexp = ri.vector([1.0 + 1j], ri.RTYPES.CPLXSXP)
    isComplex = ri.globalenv.find('is.complex')
    assert isComplex(sexp)[0]