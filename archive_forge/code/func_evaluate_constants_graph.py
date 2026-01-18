import numpy as np
from ..sharing import to_backend_cache_wrap
def evaluate_constants_graph(const_arrays, expr):
    """Convert constant arguments to tensorflow constants, and perform any
    possible constant contractions. Requires evaluating a tensorflow graph.
    """
    tf, _, _ = _get_tensorflow_and_device()
    const_arrays = [to_tensorflow(x, constant=True) for x in const_arrays]
    new_ops, new_contraction_list = expr(*const_arrays, backend='tensorflow', evaluate_constants=True)
    session = tf.get_default_session()
    new_consts = iter(session.run([x for x in new_ops if x is not None]))
    new_ops = [None if x is None else to_tensorflow(next(new_consts), constant=True) for x in new_ops]
    return (new_ops, new_contraction_list)