import builtins
from itertools import islice
from _thread import get_ident
def repr_array(self, x, level):
    if not x:
        return "array('%s')" % x.typecode
    header = "array('%s', [" % x.typecode
    return self._repr_iterable(x, level, header, '])', self.maxarray)