import subprocess
import pytest
import sys
import os
import textwrap
import rpy2.rinterface as ri
def test_complex():
    sexp = ri.ComplexSexpVector([1 + 2j])
    is_complex = ri.globalenv.find('is.complex')
    assert is_complex(sexp)[0]