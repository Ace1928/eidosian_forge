from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
import gzip
import os
import io
from petl.test.helpers import ieq, eq_
from petl.io.text import fromtext, totext
def test_fromtext_gz():
    f = NamedTemporaryFile(delete=False)
    f.close()
    fn = f.name + '.gz'
    os.rename(f.name, fn)
    f = gzip.open(fn, 'wb')
    try:
        f.write(b'foo\tbar\n')
        f.write(b'a\t1\n')
        f.write(b'b\t2\n')
        f.write(b'c\t3\n')
    finally:
        f.close()
    actual = fromtext(fn, encoding='ascii')
    expect = (('lines',), ('foo\tbar',), ('a\t1',), ('b\t2',), ('c\t3',))
    ieq(expect, actual)
    ieq(expect, actual)