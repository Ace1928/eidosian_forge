import warnings
from contextlib import contextmanager
from collections import defaultdict
from .util import subvals, toposort
from .wrap_util import wraps
def find_top_boxed_args(args):
    top_trace = -1
    top_boxes = []
    top_node_type = None
    for argnum, arg in enumerate(args):
        if isbox(arg):
            trace = arg._trace
            if trace > top_trace:
                top_boxes = [(argnum, arg)]
                top_trace = trace
                top_node_type = type(arg._node)
            elif trace == top_trace:
                top_boxes.append((argnum, arg))
    return (top_boxes, top_trace, top_node_type)