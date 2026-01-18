import collections
import hashlib
import os
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tensor_tracer_pb2
def sort_tensors_and_ops(graph):
    """Returns a wrapper that has consistent tensor and op orders."""
    graph_wrapper = collections.namedtuple('GraphWrapper', ['graph', 'operations', 'op_to_idx', 'tensors', 'tensor_to_idx', 'contains_cycle', 'topological_order_or_cycle'])
    contains_cycle, topological_order_or_cycle = topological_sort(graph)
    if not contains_cycle:
        operations = topological_order_or_cycle
    else:
        operations = graph.get_operations()
    op_to_idx = {op.name: index for index, op in enumerate(operations)}
    tensors = []
    for op in operations:
        tensors.extend(op.outputs)
    tensor_to_idx = {tensor.name: index for index, tensor in enumerate(tensors)}
    return graph_wrapper(graph=graph, operations=operations, op_to_idx=op_to_idx, tensors=tensors, tensor_to_idx=tensor_to_idx, contains_cycle=contains_cycle, topological_order_or_cycle=topological_order_or_cycle)