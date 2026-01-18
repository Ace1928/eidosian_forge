import numpy as np
from ..sharing import to_backend_cache_wrap
def cupy_contract(*arrays):
    return expr._contract([to_cupy(x) for x in arrays], backend='cupy').get()