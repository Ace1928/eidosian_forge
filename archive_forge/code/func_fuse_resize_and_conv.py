import collections
import math
import re
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import flags as flags_lib
from tensorflow.python.platform import tf_logging
from tensorflow.python.tools import strip_unused_lib
def fuse_resize_and_conv(input_graph_def, output_node_names):
    """Merges preceding resize and mirror pad ops into a specialized convolution.

  There's a common pattern of enlarging the input to a convolution using a
  resize operation, and also using MirrorPad to extend the boundaries to that
  zero edge pixels don't bleed inwards when convolving. This routine looks for
  that pattern of operations, and fuses them together into a Conv2DWithResizeOp.

  Args:
    input_graph_def: A GraphDef containing a model.
    output_node_names: A list of names of the nodes that produce the final
      results.

  Returns:
    Modified graph with resize and pad ops merged.

  Raises:
    ValueError: If the graph is badly formed with duplicate node names.
  """
    input_node_map = {}
    for node in input_graph_def.node:
        if node.name not in input_node_map:
            input_node_map[node.name] = node
        else:
            raise ValueError('Duplicate node names detected for ', node.name)
    node_reference_count = collections.defaultdict(int)
    for node in input_graph_def.node:
        for input_name in node.input:
            stripped_name = node_name_from_input(input_name)
            node_reference_count[stripped_name] += 1
    for output_name in output_node_names:
        node_reference_count[output_name] += 1
    new_ops = []
    for node in input_graph_def.node:
        if node.op != 'Conv2D':
            continue
        conv_op = node
        input_op = node_from_map(input_node_map, conv_op.input[0])
        if input_op.op == 'MirrorPad':
            mirror_pad_op = input_op
            resize_op = node_from_map(input_node_map, mirror_pad_op.input[0])
            if resize_op.op != 'ResizeBilinear':
                resize_op = None
        else:
            mirror_pad_op = None
            if input_op.op == 'ResizeBilinear':
                resize_op = input_op
            else:
                resize_op = None
        if not mirror_pad_op and (not resize_op):
            continue
        node_reference_count[conv_op.name] = 0
        if mirror_pad_op:
            node_reference_count[mirror_pad_op.name] -= 1
        if resize_op:
            node_reference_count[resize_op.name] -= 1
        fused_conv_op = node_def_pb2.NodeDef()
        if resize_op:
            fused_conv_op.op = 'FusedResizeAndPadConv2D'
        else:
            fused_conv_op.op = 'FusedPadConv2D'
        fused_conv_op.name = conv_op.name
        if mirror_pad_op:
            mirror_paddings_name = mirror_pad_op.input[1]
            mirror_paddings_mode = mirror_pad_op.attr['mode']
        else:
            paddings_op = node_def_pb2.NodeDef()
            paddings_op.op = 'Const'
            paddings_op.name = conv_op.name + '_dummy_paddings'
            paddings_op.attr['dtype'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.int32.as_datatype_enum))
            paddings_op.attr['value'].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto([0, 0, 0, 0, 0, 0, 0, 0], dtypes.int32, [4, 2])))
            new_ops.extend([paddings_op])
            mirror_paddings_name = paddings_op.name
            mirror_paddings_mode = attr_value_pb2.AttrValue(s=b'REFLECT')
        if resize_op:
            fused_conv_op.input.extend([resize_op.input[0], resize_op.input[1], mirror_paddings_name, conv_op.input[1]])
            fused_conv_op.attr['resize_align_corners'].CopyFrom(resize_op.attr['align_corners'])
        else:
            fused_conv_op.input.extend([mirror_pad_op.input[0], mirror_paddings_name, conv_op.input[1]])
        fused_conv_op.attr['T'].CopyFrom(conv_op.attr['T'])
        fused_conv_op.attr['mode'].CopyFrom(mirror_paddings_mode)
        fused_conv_op.attr['strides'].CopyFrom(conv_op.attr['strides'])
        fused_conv_op.attr['padding'].CopyFrom(conv_op.attr['padding'])
        new_ops.extend([fused_conv_op])
    result_graph_def = graph_pb2.GraphDef()
    for node in input_graph_def.node:
        if node_reference_count[node.name] < 1:
            continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        result_graph_def.node.extend([new_node])
    result_graph_def.node.extend(new_ops)
    return result_graph_def