from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.tensorflow_stub import dtypes
def keras_model_to_graph_def(keras_layer):
    """Returns a GraphDef representation of the Keras model in a dict form.

    Note that it only supports models that implemented to_json().

    Args:
      keras_layer: A dict from Keras model.to_json().

    Returns:
      A GraphDef representation of the layers in the model.
    """
    input_to_layer = {}
    model_name_to_output = {}
    g = GraphDef()
    prev_node_name = None
    for name_scope, layer in _walk_layers(keras_layer):
        if _is_model(layer):
            input_to_layer, model_name_to_output, prev_node_name = _update_dicts(name_scope, layer, input_to_layer, model_name_to_output, prev_node_name)
            continue
        layer_config = layer.get('config')
        node_name = _scoped_name(name_scope, layer_config.get('name'))
        node_def = g.node.add()
        node_def.name = node_name
        if layer.get('class_name') is not None:
            keras_cls_name = layer.get('class_name').encode('ascii')
            node_def.attr['keras_class'].s = keras_cls_name
        dtype_or_policy = layer_config.get('dtype')
        if dtype_or_policy is not None and (not isinstance(dtype_or_policy, dict)):
            tf_dtype = dtypes.as_dtype(layer_config.get('dtype'))
            node_def.attr['dtype'].type = tf_dtype.as_datatype_enum
        if layer.get('inbound_nodes') is not None:
            for maybe_inbound_node in layer.get('inbound_nodes'):
                inbound_nodes = _norm_to_list_of_layers(maybe_inbound_node)
                for [name, size, index, _] in inbound_nodes:
                    inbound_name = _scoped_name(name_scope, name)
                    inbound_node_names = model_name_to_output.get(inbound_name, [inbound_name])
                    input_name = inbound_node_names[index] if index < len(inbound_node_names) else inbound_node_names[-1]
                    node_def.input.append(input_name)
        elif prev_node_name is not None:
            node_def.input.append(prev_node_name)
        if node_name in input_to_layer:
            node_def.input.append(input_to_layer.get(node_name))
        prev_node_name = node_def.name
    return g