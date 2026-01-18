import subprocess
import pytest
import sys
import os
import textwrap
import rpy2.rinterface as ri
def test_int():
    sexp = ri.IntSexpVector([1])
    isInteger = ri.globalenv.find('is.integer')
    assert isInteger(sexp)[0]