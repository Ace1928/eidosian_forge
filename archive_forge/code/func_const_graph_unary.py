from itertools import repeat
from autograd.wrap_util import wraps
from autograd.util import subvals, toposort
from autograd.tracer import trace, Node
from functools import partial
def const_graph_unary(fun):
    graph = []
    _fun = [fun]

    def maybe_cached_fun(x):
        if graph:
            _graph = graph[0]
            vals = {_graph[0]: x}
            for node in _graph[1:]:
                vals[node] = node.partial_fun([vals[p] for p in node.parents])
            return vals[node]
        else:
            start_node = ConstGraphNode.new_root()
            end_value, end_node = trace(start_node, _fun.pop(), x)
            if end_node is None:
                raise Exception('Output is independent of input')
            graph.append(list(toposort(end_node))[::-1])
            return end_value
    return maybe_cached_fun