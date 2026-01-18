from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
from petl.compat import pickle
from petl.test.helpers import ieq
from petl.io.pickle import frompickle, topickle, appendpickle
def test_frompickle():
    f = NamedTemporaryFile(delete=False)
    table = (('foo', 'bar'), ('a', 1), ('b', 2), ('c', 2))
    for row in table:
        pickle.dump(row, f)
    f.close()
    actual = frompickle(f.name)
    ieq(table, actual)
    ieq(table, actual)