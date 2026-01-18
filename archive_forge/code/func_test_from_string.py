import subprocess
import pytest
import sys
import os
import textwrap
import rpy2.rinterface as ri
def test_from_string():
    sexp = ri.vector(['abc'], ri.RTYPES.STRSXP)
    isCharacter = ri.globalenv.find('is.character')
    assert isCharacter(sexp)[0]