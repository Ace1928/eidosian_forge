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
def model_to_graph_def(model, **kwargs):
    """
    Convert a Caffe2 model to a Tensorflow graph.

    This function extracts 'param_init_net' and 'net' from the model and passes it to nets_to_graph()
    for further processing.

    Args:
        model (cnn.CNNModelHelper, model_helper.ModelHelper): The model to
            extract the nets (instances of core.Net) from.

    Returns:
        Call to nets_to_graph_def() with extracted 'param_init_net', 'net' and
            **kwargs. See _operators_to_graph_def for detailed **kwargs.
    """
    nets = [model.param_init_net, model.net]
    return nets_to_graph_def(nets, **kwargs)