import subprocess
import pytest
import sys
import os
import textwrap
import rpy2.rinterface as ri
def test_bool():
    sexp = ri.BoolSexpVector([True])
    isBool = ri.globalenv.find('is.logical')
    assert isBool(sexp)[0]