from __future__ import division, print_function, absolute_import
from itertools import chain
from tempfile import NamedTemporaryFile
import pytest
from petl.test.helpers import ieq
from petl.transform.sorts import sort
import petl as etl
from petl.io.pytables import fromhdf5, fromhdf5sorted, tohdf5, appendhdf5
def test_appendhdf5():
    f = NamedTemporaryFile()
    h5file = tables.open_file(f.name, mode='w', title='Test file')
    h5file.create_group('/', 'testgroup', 'Test Group')
    h5file.create_table('/testgroup', 'testtable', FooBar, 'Test Table')
    h5file.flush()
    h5file.close()
    table1 = (('foo', 'bar'), (1, b'asdfgh'), (2, b'qwerty'), (3, b'zxcvbn'))
    tohdf5(table1, f.name, '/testgroup', 'testtable')
    ieq(table1, fromhdf5(f.name, '/testgroup', 'testtable'))
    appendhdf5(table1, f.name, '/testgroup', 'testtable')
    ieq(chain(table1, table1[1:]), fromhdf5(f.name, '/testgroup', 'testtable'))