from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
import sqlite3
from petl.test.helpers import ieq
from petl.io.db import fromdb, todb, appenddb
def test_fromsqlite3():
    f = NamedTemporaryFile(delete=False)
    f.close()
    data = (('a', 1), ('b', 2), ('c', 2.0))
    connection = sqlite3.connect(f.name)
    c = connection.cursor()
    c.execute('CREATE TABLE foobar (foo, bar)')
    for row in data:
        c.execute('INSERT INTO foobar VALUES (?, ?)', row)
    connection.commit()
    c.close()
    connection.close()
    actual = fromdb(f.name, 'SELECT * FROM foobar')
    expect = (('foo', 'bar'), ('a', 1), ('b', 2), ('c', 2.0))
    ieq(expect, actual, cast=tuple)
    ieq(expect, actual, cast=tuple)