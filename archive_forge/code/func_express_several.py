from sage.all import (cached_method, real_part, imag_part, round, ceil, floor, log,
import itertools
def express_several(self, elts, prec=None):
    """
        Return exact expressions every number elts, provided this is
        possible.
        """
    ans = []
    for a in elts:
        exact = self.express(a, prec)
        if exact is None:
            return None
        else:
            ans.append(exact)
    return ans