from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils.generic_utils import LazyLoader
def should_use_v2():
    """Determine if v1 or v2 version should be used."""
    if context.executing_eagerly():
        return True
    elif ops.executing_eagerly_outside_functions():
        graph = ops.get_default_graph()
        if getattr(graph, 'name', False) and graph.name.startswith('wrapped_function'):
            return False
        return True
    else:
        return False