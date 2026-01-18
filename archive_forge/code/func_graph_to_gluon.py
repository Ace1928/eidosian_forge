import numpy as np
from .... import symbol
from .... import ndarray as nd
from ....base import string_types
from ._import_helper import _convert_map as convert_map
def graph_to_gluon(self, graph, ctx, opset_version):
    """Construct SymbolBlock from onnx graph.

        Parameters
        ----------
        graph : onnx protobuf object
            The loaded onnx graph
        ctx : Context or list of Context
            Loads the model into one or many context(s).

        Returns
        -------
        sym_block :gluon.nn.SymbolBlock
            The returned gluon SymbolBlock
        """
    sym, arg_params, aux_params = self.from_onnx(graph, opset_version)
    metadata = self.get_graph_metadata(graph)
    data_names = [input_tensor[0] for input_tensor in metadata['input_tensor_data']]
    data_inputs = [symbol.var(data_name) for data_name in data_names]
    from ....gluon import SymbolBlock
    net = SymbolBlock(outputs=sym, inputs=data_inputs)
    net_params = net.collect_params()
    for param in arg_params:
        if param in net_params:
            net_params[param].shape = arg_params[param].shape
            net_params[param]._load_init(arg_params[param], ctx=ctx)
    for param in aux_params:
        if param in net_params:
            net_params[param].shape = aux_params[param].shape
            net_params[param]._load_init(aux_params[param], ctx=ctx)
    return net