from numba.core.typeconv import castgraph, Conversion
from numba.core import types
def safe_unsafe(self, a, b):
    """
        Set `a` can safe convert to `b` and `b` can unsafe convert to `a`
        """
    self._tg.safe(a, b)
    self._tg.unsafe(b, a)