import numpy as np
from ..sharing import to_backend_cache_wrap
def evaluate_constants(const_arrays, expr):
    _, _, eager = _get_tensorflow_and_device()
    fn = evaluate_constants_eager if eager else evaluate_constants_graph
    return fn(const_arrays, expr)