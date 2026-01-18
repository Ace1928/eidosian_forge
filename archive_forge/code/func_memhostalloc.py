import numpy as np
from collections import namedtuple
def memhostalloc(self, sz, mapped=False, portable=False, wc=False):
    """Allocates memory on the host"""
    return self.memalloc(sz)