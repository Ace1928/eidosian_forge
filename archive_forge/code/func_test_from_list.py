import subprocess
import pytest
import sys
import os
import textwrap
import rpy2.rinterface as ri
def test_from_list():
    seq = (ri.FloatSexpVector([1.0]), ri.IntSexpVector([2, 3]), ri.StrSexpVector(['foo', 'bar']))
    sexp = ri.ListSexpVector(seq)
    isList = ri.globalenv.find('is.list')
    assert isList(sexp)[0]
    assert len(sexp) == 3