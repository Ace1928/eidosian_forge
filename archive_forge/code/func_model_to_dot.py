import os
import sys
from tensorflow.python.keras.utils.io_utils import path_to_string
from tensorflow.python.util import nest
def model_to_dot(model, show_shapes=False, show_dtype=False, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96, subgraph=False):
    """Convert a Keras model to dot format.

  Args:
    model: A Keras model instance.
    show_shapes: whether to display shape information.
    show_dtype: whether to display layer dtypes.
    show_layer_names: whether to display layer names.
    rankdir: `rankdir` argument passed to PyDot,
        a string specifying the format of the plot:
        'TB' creates a vertical plot;
        'LR' creates a horizontal plot.
    expand_nested: whether to expand nested models into clusters.
    dpi: Dots per inch.
    subgraph: whether to return a `pydot.Cluster` instance.

  Returns:
    A `pydot.Dot` instance representing the Keras model or
    a `pydot.Cluster` instance representing nested model if
    `subgraph=True`.

  Raises:
    ImportError: if graphviz or pydot are not available.
  """
    from tensorflow.python.keras.layers import wrappers
    from tensorflow.python.keras.engine import sequential
    from tensorflow.python.keras.engine import functional
    if not check_pydot():
        message = ('You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) ', 'for plot_model/model_to_dot to work.')
        if 'IPython.core.magics.namespace' in sys.modules:
            print(message)
            return
        else:
            raise ImportError(message)
    if subgraph:
        dot = pydot.Cluster(style='dashed', graph_name=model.name)
        dot.set('label', model.name)
        dot.set('labeljust', 'l')
    else:
        dot = pydot.Dot()
        dot.set('rankdir', rankdir)
        dot.set('concentrate', True)
        dot.set('dpi', dpi)
        dot.set_node_defaults(shape='record')
    sub_n_first_node = {}
    sub_n_last_node = {}
    sub_w_first_node = {}
    sub_w_last_node = {}
    layers = model.layers
    if not model._is_graph_network:
        node = pydot.Node(str(id(model)), label=model.name)
        dot.add_node(node)
        return dot
    elif isinstance(model, sequential.Sequential):
        if not model.built:
            model.build()
        layers = super(sequential.Sequential, model).layers
    for i, layer in enumerate(layers):
        layer_id = str(id(layer))
        layer_name = layer.name
        class_name = layer.__class__.__name__
        if isinstance(layer, wrappers.Wrapper):
            if expand_nested and isinstance(layer.layer, functional.Functional):
                submodel_wrapper = model_to_dot(layer.layer, show_shapes, show_dtype, show_layer_names, rankdir, expand_nested, subgraph=True)
                sub_w_nodes = submodel_wrapper.get_nodes()
                sub_w_first_node[layer.layer.name] = sub_w_nodes[0]
                sub_w_last_node[layer.layer.name] = sub_w_nodes[-1]
                dot.add_subgraph(submodel_wrapper)
            else:
                layer_name = '{}({})'.format(layer_name, layer.layer.name)
                child_class_name = layer.layer.__class__.__name__
                class_name = '{}({})'.format(class_name, child_class_name)
        if expand_nested and isinstance(layer, functional.Functional):
            submodel_not_wrapper = model_to_dot(layer, show_shapes, show_dtype, show_layer_names, rankdir, expand_nested, subgraph=True)
            sub_n_nodes = submodel_not_wrapper.get_nodes()
            sub_n_first_node[layer.name] = sub_n_nodes[0]
            sub_n_last_node[layer.name] = sub_n_nodes[-1]
            dot.add_subgraph(submodel_not_wrapper)
        if show_layer_names:
            label = '{}: {}'.format(layer_name, class_name)
        else:
            label = class_name
        if show_dtype:

            def format_dtype(dtype):
                if dtype is None:
                    return '?'
                else:
                    return str(dtype)
            label = '%s|%s' % (label, format_dtype(layer.dtype))
        if show_shapes:

            def format_shape(shape):
                return str(shape).replace(str(None), 'None')
            try:
                outputlabels = format_shape(layer.output_shape)
            except AttributeError:
                outputlabels = '?'
            if hasattr(layer, 'input_shape'):
                inputlabels = format_shape(layer.input_shape)
            elif hasattr(layer, 'input_shapes'):
                inputlabels = ', '.join([format_shape(ishape) for ishape in layer.input_shapes])
            else:
                inputlabels = '?'
            label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label, inputlabels, outputlabels)
        if not expand_nested or not isinstance(layer, functional.Functional):
            node = pydot.Node(layer_id, label=label)
            dot.add_node(node)
    for layer in layers:
        layer_id = str(id(layer))
        for i, node in enumerate(layer._inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model._network_nodes:
                for inbound_layer in nest.flatten(node.inbound_layers):
                    inbound_layer_id = str(id(inbound_layer))
                    if not expand_nested:
                        assert dot.get_node(inbound_layer_id)
                        assert dot.get_node(layer_id)
                        add_edge(dot, inbound_layer_id, layer_id)
                    elif not isinstance(inbound_layer, functional.Functional) and (not is_wrapped_model(inbound_layer)):
                        if not isinstance(layer, functional.Functional) and (not is_wrapped_model(layer)):
                            assert dot.get_node(inbound_layer_id)
                            assert dot.get_node(layer_id)
                            add_edge(dot, inbound_layer_id, layer_id)
                        elif isinstance(layer, functional.Functional):
                            add_edge(dot, inbound_layer_id, sub_n_first_node[layer.name].get_name())
                        elif is_wrapped_model(layer):
                            add_edge(dot, inbound_layer_id, layer_id)
                            name = sub_w_first_node[layer.layer.name].get_name()
                            add_edge(dot, layer_id, name)
                    elif isinstance(inbound_layer, functional.Functional):
                        name = sub_n_last_node[inbound_layer.name].get_name()
                        if isinstance(layer, functional.Functional):
                            output_name = sub_n_first_node[layer.name].get_name()
                            add_edge(dot, name, output_name)
                        else:
                            add_edge(dot, name, layer_id)
                    elif is_wrapped_model(inbound_layer):
                        inbound_layer_name = inbound_layer.layer.name
                        add_edge(dot, sub_w_last_node[inbound_layer_name].get_name(), layer_id)
    return dot