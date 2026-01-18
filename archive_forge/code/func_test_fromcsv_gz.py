from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
import gzip
import os
import logging
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
from petl.io.csv import fromcsv, fromtsv, tocsv, appendcsv, totsv, appendtsv
def test_fromcsv_gz():
    data = [b'foo,bar', b'a,1', b'b,2', b'c,2']
    expect = (('foo', 'bar'), ('a', '1'), ('b', '2'), ('c', '2'))
    if PY2:
        lts = (b'\n', b'\r\n')
    else:
        lts = (b'\r', b'\n', b'\r\n')
    for lt in lts:
        f = NamedTemporaryFile(delete=False)
        f.close()
        fn = f.name + '.gz'
        os.rename(f.name, fn)
        fz = gzip.open(fn, 'wb')
        fz.write(lt.join(data))
        fz.close()
        actual = fromcsv(fn, encoding='ascii')
        ieq(expect, actual)
        ieq(expect, actual)