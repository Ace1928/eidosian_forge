from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
import gzip
import os
import io
from petl.test.helpers import ieq, eq_
from petl.io.text import fromtext, totext
def test_fromtext_lineterminators():
    data = [b'foo,bar', b'a,1', b'b,2', b'c,2']
    expect = (('lines',), ('foo,bar',), ('a,1',), ('b,2',), ('c,2',))
    for lt in (b'\r', b'\n', b'\r\n'):
        f = NamedTemporaryFile(mode='wb', delete=False)
        f.write(lt.join(data))
        f.close()
        actual = fromtext(f.name, encoding='ascii')
        ieq(expect, actual)