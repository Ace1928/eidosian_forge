import subprocess
import pytest
import sys
import os
import textwrap
import rpy2.rinterface as ri
def test_byte():
    seq = (b'a', b'b')
    sexp = ri.ByteSexpVector(seq)
    is_raw = ri.globalenv.find('is.raw')
    assert is_raw(sexp)[0]