from __future__ import absolute_import, print_function, division
import gzip
import bz2
import zipfile
from tempfile import NamedTemporaryFile
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
import petl as etl
from petl.io.sources import MemorySource, PopenSource, ZipSource, \
def test_memorysource():
    tbl1 = (('foo', 'bar'), ('a', '1'), ('b', '2'), ('c', '2'))
    ss = MemorySource()
    etl.tocsv(tbl1, ss)
    expect = 'foo,bar\r\na,1\r\nb,2\r\nc,2\r\n'
    if not PY2:
        expect = expect.encode('ascii')
    actual = ss.getvalue()
    eq_(expect, actual)
    tbl2 = etl.fromcsv(MemorySource(actual))
    ieq(tbl1, tbl2)
    ieq(tbl1, tbl2)
    etl.appendcsv(tbl1, ss)
    actual = ss.getvalue()
    expect = 'foo,bar\r\na,1\r\nb,2\r\nc,2\r\na,1\r\nb,2\r\nc,2\r\n'
    if not PY2:
        expect = expect.encode('ascii')
    eq_(expect, actual)