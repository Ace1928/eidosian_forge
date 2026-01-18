from __future__ import absolute_import, print_function, division
import sqlite3
from tempfile import NamedTemporaryFile
from petl.compat import next
from petl.test.helpers import ieq, eq_
from petl.io.db import fromdb, todb, appenddb
def test_fromdb_withargs():
    data = (('a', 1), ('b', 2), ('c', 2.0))
    connection = sqlite3.connect(':memory:')
    c = connection.cursor()
    c.execute('create table foobar (foo, bar)')
    for row in data:
        c.execute('insert into foobar values (?, ?)', row)
    connection.commit()
    c.close()
    actual = fromdb(connection, 'select * from foobar where bar > ? and bar < ?', (1, 3))
    expect = (('foo', 'bar'), ('b', 2), ('c', 2.0))
    ieq(expect, actual)
    ieq(expect, actual)