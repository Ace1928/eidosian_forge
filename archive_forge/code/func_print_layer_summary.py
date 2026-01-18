import functools
import weakref
import numpy as np
from tensorflow.python.util import nest
def print_layer_summary(layer):
    """Prints a summary for a single layer.

    Args:
        layer: target layer.
    """
    try:
        output_shape = layer.output_shape
    except AttributeError:
        output_shape = 'multiple'
    except RuntimeError:
        output_shape = '?'
    name = layer.name
    cls_name = layer.__class__.__name__
    if not layer.built and (not getattr(layer, '_is_graph_network', False)):
        params = '0 (unused)'
    else:
        params = layer.count_params()
    fields = [name + ' (' + cls_name + ')', output_shape, params]
    print_row(fields, positions)