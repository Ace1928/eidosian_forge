import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import input_layer as input_layer_module
from keras.src.engine import keras_tensor
from keras.src.engine import node as node_module
def find_nodes_by_inputs_and_outputs(inputs, outputs):
    """Fetch all Nodes in the graph defined by "inputs" and "outputs".

    This method is used to find and then clone Nodes when creating a new
    sub-model from an existing functional model.

    Args:
      inputs: A nested structure of KerasTensor to use as model inputs.
      outputs: A nested structure of KerasTensor to use as model outputs.

    Returns:
      A list of Nodes that are connected to the inputs and outputs.

    Raises:
      ValueError: when inputs and outputs are disconnected or in case of
        unexpected objects in the inputs/outputs.
    """
    start_keras_tensors = tf.nest.flatten(outputs)
    end_keras_tensors = tf.nest.flatten(inputs)
    for t in start_keras_tensors + end_keras_tensors:
        if not node_module.is_keras_tensor(t):
            raise ValueError(_KERAS_TENSOR_TYPE_CHECK_ERROR_MSG.format(t))
    end_ids = set([id(kt) for kt in end_keras_tensors])
    end_ids_found = set()
    nodes_to_visit = []
    nodes_in_graph = []
    node_id_visited = set()
    for t in start_keras_tensors:
        nodes_to_visit.append(t.node)
    while nodes_to_visit:
        node = nodes_to_visit.pop(0)
        if id(node) in node_id_visited:
            continue
        node_id_visited.add(id(node))
        nodes_in_graph.append(node)
        for kt in node.keras_inputs:
            if id(kt) in end_ids:
                end_ids_found.add(id(kt))
                continue
            inbound_node = kt.node
            if inbound_node.is_input:
                raise ValueError('Found input tensor cannot be reached given provided output tensors. Please make sure the tensor {} is included in the model inputs when building functional model.'.format(kt))
            nodes_to_visit.append(inbound_node)
    if end_ids != end_ids_found:
        unvisited_inputs = [kt for kt in end_keras_tensors if id(kt) not in end_ids_found]
        raise ValueError('Found unvisited input tensors that are disconnected from the outputs: {}'.format(unvisited_inputs))
    return nodes_in_graph