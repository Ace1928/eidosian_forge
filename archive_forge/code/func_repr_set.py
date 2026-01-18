import builtins
from itertools import islice
from _thread import get_ident
def repr_set(self, x, level):
    if not x:
        return 'set()'
    x = _possibly_sorted(x)
    return self._repr_iterable(x, level, '{', '}', self.maxset)