import collections
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.saver import export_meta_graph
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
def convert_variable_to_constant(self, incoming_edge, tensor_data):
    super(_While, self).convert_variable_to_constant(incoming_edge, tensor_data)
    node = self.converted_self()
    if node.node.attr['output_shapes'].list.shape:
        node.node.attr['output_shapes'].list.shape[incoming_edge.destination.index].CopyFrom(tensor_shape_pb2.TensorShapeProto(dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=dim) for dim in tensor_data.numpy.shape]))
    body_name = self._node.attr['body'].func.name
    body = self._enclosing_graph.functions[body_name].converted_self().function
    body.signature.output_arg[incoming_edge.destination.index].type = tensor_data.dtype