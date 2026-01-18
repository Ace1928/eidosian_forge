import weakref
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def prune(self, feeds, fetches, name=None, input_signature=None):
    """Extract a subgraph of this function's underlying graph.

    Wraps the subgraph in a new `WrappedFunction` object.

    Args:
      feeds: Input tensors to the subgraph to extract, as `Tensor` objects.
      fetches: Possibly-nested Python data structure containing information
        about outputs of the target subgraph. Each entry can either be a
        `Tensor` object (for data outputs), an `Operation` object (for control
        outputs), or a `TensorInfo` proto. Any additional shape/dtype
        information provided in a `TensorInfo` and not present in the original
        graph will be added to the returned subgraph.
      name: (optional) Name to give to the underlying `FuncGraph` of the
        returned object. If no name is provided, the graph's name will be
        `"pruned"`.
      input_signature: (optional) possibly-nested Python data structure
        containing `TensorSpec` objects, with which to populate the returned
        functions's `FuncGraph`'s `structured_input_signature` field.

    Returns:
      A new `WrappedFunction` object containing a copy of the portion of this
        object's graph that goes from `feeds` to `fetches`.
    """
    name = name or 'pruned'
    flat_feeds = nest.flatten(feeds, expand_composites=True)
    flat_feeds = [self.graph.as_graph_element(t) for t in flat_feeds]
    for f in flat_feeds:
        if not isinstance(f, tensor_lib.Tensor):
            raise ValueError(f'All memebers of argument `feeds` must be tensors. Got {f} with type {type(f)}.')
    internal_captures = {id(c) for c in self.graph.internal_captures}
    flat_feeds = [f for f in flat_feeds if id(f) not in internal_captures]
    operation_fetches = []
    tensor_fetches = []
    tensor_infos = []

    def _fetch_preprocessing_callback(fetch):
        """Extract out lists of ops, tensors, and tensor type info.

      Turns TensorInfos into Tensors in the original `fetches` structure.
      Also extracts ops from `fetches`.

      Args:
        fetch: The fetch to preprocess: Tensor, TensorInfo, or Operation, or
          string identifying a Tensor or Operation.

      Returns:
        `fetch` converted to a Tensor.
      """
        if isinstance(fetch, ops.Operation):
            operation_fetches.append(fetch)
            return fetch
        elif isinstance(fetch, meta_graph_pb2.TensorInfo):
            tensor_infos.append(fetch)
            decoded = _get_element_from_tensor_info(fetch, self._func_graph)
            if tensor_util.is_tf_type(decoded) or isinstance(decoded, composite_tensor.CompositeTensor):
                tensor_fetches.append(decoded)
            else:
                operation_fetches.append(decoded)
            return decoded
        elif isinstance(fetch, (tensor_lib.Tensor, composite_tensor.CompositeTensor)):
            tensor_fetches.append(fetch)
            return fetch
        else:
            graph_element = self.graph.as_graph_element(fetch)
            return _fetch_preprocessing_callback(graph_element)
    fetches = nest.map_structure(_fetch_preprocessing_callback, fetches)
    tensor_fetches = nest.flatten(tensor_fetches, expand_composites=True)
    for f in flat_feeds + tensor_fetches + operation_fetches:
        if f.graph is not self._func_graph:
            raise ValueError(f'Can only prune function whose feeds and fetches from graph {self._func_graph}. Input {f} is from a different graph {f.graph}.')
    with self._func_graph.as_default():
        pruned_graph = func_graph.FuncGraph(name)
    lift_map = lift_to_graph.lift_to_graph(operation_fetches + tensor_fetches, pruned_graph, sources=flat_feeds + self.graph.internal_captures, base_graph=self._func_graph)
    pruned_graph.outputs.extend((lift_map[x] for x in tensor_fetches))
    pruned_graph.control_outputs.extend([lift_map[operation] for operation in operation_fetches])
    pruned_graph.inputs.extend((lift_map[x] for x in flat_feeds))
    for external_capture, internal_capture in self.graph.captures:
        pruned_graph.add_capture(external_capture, lift_map[internal_capture])
    for ti in tensor_infos:
        if ti.WhichOneof('encoding') == 'name':
            t = pruned_graph.as_graph_element(ti.name)
            if tensor_util.is_tf_type(t):
                t.set_shape(tensor_shape.TensorShape(ti.tensor_shape))
    for f in self.graph._functions.values():
        pruned_graph._add_function(f)
    pruned_graph.variables = self.graph.variables

    def _structured_output_mapping(fetched):
        """callback for `nest.map_structure()`"""
        lifted = lift_map[fetched]
        if isinstance(lifted, ops.Operation):
            return None
        return lifted
    pruned_graph.structured_outputs = nest.map_structure(_structured_output_mapping, fetches, expand_composites=True)
    if input_signature:
        args, kwargs = input_signature
        args = () if args is None else args
        input_signature = (args, kwargs)
    pruned_graph.structured_input_signature = input_signature
    pruned_fn = WrappedFunction(pruned_graph, variable_holder=self._variable_holder)
    pruned_fn._num_positional_args = len(flat_feeds)
    pruned_fn._arg_keywords = [tensor.op.name for tensor in flat_feeds]
    return pruned_fn