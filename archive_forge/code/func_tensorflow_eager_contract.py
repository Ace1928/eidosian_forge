import numpy as np
from ..sharing import to_backend_cache_wrap
def tensorflow_eager_contract(*arrays):
    return expr._contract([to_tensorflow(x) for x in arrays], backend='tensorflow').numpy()