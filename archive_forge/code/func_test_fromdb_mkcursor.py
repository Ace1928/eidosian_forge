from __future__ import absolute_import, print_function, division
import sqlite3
from tempfile import NamedTemporaryFile
from petl.compat import next
from petl.test.helpers import ieq, eq_
from petl.io.db import fromdb, todb, appenddb
def test_fromdb_mkcursor():
    data = (('a', 1), ('b', 2), ('c', 2.0))
    connection = sqlite3.connect(':memory:')
    c = connection.cursor()
    c.execute('create table foobar (foo, bar)')
    for row in data:
        c.execute('insert into foobar values (?, ?)', row)
    connection.commit()
    c.close()
    mkcursor = lambda: connection.cursor()
    actual = fromdb(mkcursor, 'select * from foobar')
    expect = (('foo', 'bar'), ('a', 1), ('b', 2), ('c', 2.0))
    ieq(expect, actual)
    ieq(expect, actual)
    i1 = iter(actual)
    i2 = iter(actual)
    eq_(('foo', 'bar'), next(i1))
    eq_(('a', 1), next(i1))
    eq_(('foo', 'bar'), next(i2))
    eq_(('b', 2), next(i1))