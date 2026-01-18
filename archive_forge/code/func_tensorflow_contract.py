import numpy as np
from ..sharing import to_backend_cache_wrap
def tensorflow_contract(*arrays):
    session = tf.get_default_session()
    feed_dict = {p: a for p, a in zip(placeholders, arrays) if p.op.type == 'Placeholder'}
    return session.run(graph, feed_dict=feed_dict)