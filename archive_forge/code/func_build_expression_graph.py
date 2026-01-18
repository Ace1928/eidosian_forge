import numpy as np
from ..sharing import to_backend_cache_wrap
def build_expression_graph(arrays, expr):
    """Build a tensorflow function based on ``arrays`` and ``expr``.
    """
    tf, _, _ = _get_tensorflow_and_device()
    placeholders = [to_tensorflow(array) for array in arrays]
    graph = expr._contract(placeholders, backend='tensorflow')

    def tensorflow_contract(*arrays):
        session = tf.get_default_session()
        feed_dict = {p: a for p, a in zip(placeholders, arrays) if p.op.type == 'Placeholder'}
        return session.run(graph, feed_dict=feed_dict)
    return tensorflow_contract