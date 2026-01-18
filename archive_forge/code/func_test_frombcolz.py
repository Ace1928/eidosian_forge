from __future__ import absolute_import, print_function, division
import tempfile
import pytest
from petl.test.helpers import ieq, eq_
from petl.io.bcolz import frombcolz, tobcolz, appendbcolz
def test_frombcolz():
    cols = [['apples', 'oranges', 'pears'], [1, 3, 7], [2.5, 4.4, 0.1]]
    names = ('foo', 'bar', 'baz')
    rootdir = tempfile.mkdtemp()
    ctbl = bcolz.ctable(cols, names=names, rootdir=rootdir, mode='w')
    ctbl.flush()
    expect = [names] + list(zip(*cols))
    actual = frombcolz(ctbl)
    ieq(expect, actual)
    ieq(expect, actual)
    actual = frombcolz(rootdir)
    ieq(expect, actual)
    ieq(expect, actual)