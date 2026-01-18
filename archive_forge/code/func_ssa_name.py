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
def ssa_name(name: str, versions: Dict[str, int]) -> int:
    assert name in versions
    version = versions[name]
    if (name, version) in versioned:
        return versioned[name, version]
    new_name = _make_unique_name(seen, name, min_version=version)
    versioned[name, version] = new_name
    if name in shapes:
        new_shapes[new_name] = shapes[name]
    if blob_name_tracker and name in blob_name_tracker:
        new_blob_name_tracker[new_name] = blob_name_tracker[name]
    return new_name