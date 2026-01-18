import functools
import operator
from functools import reduce
from typing import Any, Tuple
import torch
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from .. import ir
from ..lowering import lowerings as L
from ..pattern_matcher import (
from ..virtualized import ops
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern
from .quantization import (
@register_freezing_graph_pattern(CallFunction(aten.mkldnn_rnn_layer.default, *_aten_mkldnn_rnn_layer_args), extra_check=_is_packable_mkldnn_rnn_layer)
def mkldnn_rnn_layer(match, *args, **kwargs):

    def get_item(graph, node, index):
        return graph.call_function(operator.getitem, (node, index))
    graph = match.graph
    lstm_node = match.output_node()
    input = args[0]
    weight0, weight1 = args[1:3]
    reverse = kwargs.get('reverse')
    packed_lstm_op = aten.mkldnn_rnn_layer.default
    hidden_size = args[9]
    has_biases = args[11]
    batch_first = args[13]
    with graph.inserting_before(lstm_node):
        packed_weight_op = mkldnn._reorder_mkldnn_rnn_layer_weight.default
        packed_weight_inputs = (weight0, weight1, hidden_size, reverse, has_biases, batch_first)
        packed_weight_node = graph.create_node('call_function', packed_weight_op, packed_weight_inputs, {}, 'name')
        packed_weight_items = [get_item(graph, packed_weight_node, i) for i in range(2)]
        pack_lstm_inputs = (args[0], *packed_weight_items, args[3], args[4], args[5], args[6], reverse, *args[7:])
        packed_lstm_node = graph.create_node('call_function', packed_lstm_op, args=pack_lstm_inputs)
        lstm_node.replace_all_uses_with(packed_lstm_node)
        packed_lstm_node.meta.update(lstm_node.meta)
        graph.erase_node(lstm_node)