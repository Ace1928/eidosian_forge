from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
import gzip
import os
import logging
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
from petl.io.csv import fromcsv, fromtsv, tocsv, appendcsv, totsv, appendtsv
def test_fromcsv_lineterminators():
    data = [b'foo,bar', b'a,1', b'b,2', b'c,2']
    expect = (('foo', 'bar'), ('a', '1'), ('b', '2'), ('c', '2'))
    for lt in (b'\r', b'\n', b'\r\n'):
        debug(repr(lt))
        f = NamedTemporaryFile(mode='wb', delete=False)
        f.write(lt.join(data))
        f.close()
        with open(f.name, 'rb') as g:
            debug(repr(g.read()))
        actual = fromcsv(f.name, encoding='ascii')
        debug(actual)
        ieq(expect, actual)