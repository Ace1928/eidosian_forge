from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import dtypes
def swap_tensor_content_in_graph_node(graph_def, from_endiness, to_endiness):
    for node in graph_def.node:
        if node.op == 'Const':
            tensor = node.attr['value'].tensor
            byte_swap_tensor_content(tensor, from_endiness, to_endiness)