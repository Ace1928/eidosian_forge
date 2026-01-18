import builtins
from itertools import islice
from _thread import get_ident
def repr_deque(self, x, level):
    return self._repr_iterable(x, level, 'deque([', '])', self.maxdeque)