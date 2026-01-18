from itertools import repeat
from autograd.wrap_util import wraps
from autograd.util import subvals, toposort
from autograd.tracer import trace, Node
from functools import partial
def full_graph(fun, *args, **kwargs):
    unary_fun = lambda args: fun(*args, **kwargs)
    start_node = FullGraphNode.new_root()
    end_value, end_node = trace(start_node, unary_fun, args)
    return end_node