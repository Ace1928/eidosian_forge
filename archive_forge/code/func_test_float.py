import subprocess
import pytest
import sys
import os
import textwrap
import rpy2.rinterface as ri
def test_float():
    sexp = ri.IntSexpVector([1.0])
    isNumeric = ri.globalenv.find('is.numeric')
    assert isNumeric(sexp)[0]