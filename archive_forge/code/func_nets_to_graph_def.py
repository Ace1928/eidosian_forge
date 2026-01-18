import copy
import logging
import os
import re
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from typing import Set, Dict, Tuple, List
def nets_to_graph_def(nets, shapes=None, **kwargs):
    """
    Convert a set of Caffe2 nets to a Tensorflow graph.

    Args:
        nets: List of core.Nets. core.Net is a wrapper around a NetDef protobuf.
            The corresponding protobuf can be extracted using .Proto().
        shapes: Dictionary mapping blob names to their shapes/dimensions.

    Returns:
        Call to protos_to_graph_def() with the extracted NetDef protobufs and
            **kwargs. See _operators_to_graph_def for detailed **kwargs.
    """
    shapes = {}
    nets = [copy.deepcopy(net.Proto()) for net in nets]
    shapes = copy.deepcopy(shapes)
    return protos_to_graph_def(nets, shapes, **kwargs)