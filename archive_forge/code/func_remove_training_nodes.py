import copy
import re
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import _proto_comparators
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(date=None, instructions=_DEPRECATION_MSG)
@tf_export(v1=['graph_util.remove_training_nodes'])
def remove_training_nodes(input_graph, protected_nodes=None):
    """Prunes out nodes that aren't needed for inference.

  There are nodes like Identity and CheckNumerics that are only useful
  during training, and can be removed in graphs that will be used for
  nothing but inference. Here we identify and remove them, returning an
  equivalent graph. To be specific, CheckNumerics nodes are always removed, and
  Identity nodes that aren't involved in control edges are spliced out so that
  their input and outputs are directly connected.

  Args:
    input_graph: Model to analyze and prune.
    protected_nodes: An optional list of names of nodes to be kept
      unconditionally. This is for example useful to preserve Identity output
      nodes.

  Returns:
    A list of nodes with the unnecessary ones removed.
  """
    if not protected_nodes:
        protected_nodes = []
    types_to_remove = {'CheckNumerics': True}
    input_nodes = input_graph.node
    names_to_remove = {}
    for node in input_nodes:
        if node.op in types_to_remove and node.name not in protected_nodes:
            names_to_remove[node.name] = True
    nodes_after_removal = []
    for node in input_nodes:
        if node.name in names_to_remove:
            continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        input_before_removal = node.input
        del new_node.input[:]
        for full_input_name in input_before_removal:
            input_name = re.sub('^\\^', '', full_input_name)
            if input_name in names_to_remove:
                continue
            new_node.input.append(full_input_name)
        nodes_after_removal.append(new_node)
    types_to_splice = {'Identity': True}
    control_input_names = set()
    node_names_with_control_input = set()
    node_in_colocated = set()
    for node in nodes_after_removal:
        for node_input in node.input:
            if '^' in node_input:
                control_input_names.add(node_input.replace('^', ''))
                node_names_with_control_input.add(node.name)
        if '_class' in node.attr:
            for colocated_node_name in node.attr['_class'].list.s:
                node_in_colocated.add(_get_colocated_node_name(colocated_node_name))
    names_to_splice = {}
    for node in nodes_after_removal:
        if node.op in types_to_splice and node.name not in protected_nodes:
            if node.name in node_in_colocated:
                continue
            if node.name not in node_names_with_control_input:
                names_to_splice[node.name] = node.input[0]
    names_to_splice = {name: value for name, value in names_to_splice.items() if name not in control_input_names}
    nodes_after_splicing = []
    for node in nodes_after_removal:
        if node.name in names_to_splice:
            continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        input_before_removal = node.input
        del new_node.input[:]
        for full_input_name in input_before_removal:
            input_name = re.sub('^\\^', '', full_input_name)
            while input_name in names_to_splice:
                full_input_name = names_to_splice[input_name]
                input_name = re.sub('^\\^', '', full_input_name)
            new_node.input.append(full_input_name)
        nodes_after_splicing.append(new_node)
    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(nodes_after_splicing)
    return output_graph