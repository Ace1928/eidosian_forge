from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
import gzip
import os
import logging
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
from petl.io.csv import fromcsv, fromtsv, tocsv, appendcsv, totsv, appendtsv
def test_fromtsv():
    data = [b'foo\tbar', b'a\t1', b'b\t2', b'c\t2']
    f = NamedTemporaryFile(mode='wb', delete=False)
    f.write(b'\n'.join(data))
    f.close()
    expect = (('foo', 'bar'), ('a', '1'), ('b', '2'), ('c', '2'))
    actual = fromtsv(f.name, encoding='ascii')
    ieq(expect, actual)
    ieq(expect, actual)