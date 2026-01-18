import logging
import numpy as np
from mxnet.base import string_types
from mxnet import symbol
from ._export_onnx import MXNetGraph
from ._export_helper import load_module
def get_operator_support(opset_version=None):
    """Return a list of MXNet operators supported by the current/specified opset
    """
    try:
        from onnx.defs import onnx_opset_version
    except ImportError:
        raise ImportError('Onnx and protobuf need to be installed. ' + 'Instructions to install - https://github.com/onnx/onnx')
    if opset_version is None:
        opset_version = onnx_opset_version()
    all_versions = range(opset_version, 11, -1)
    ops = set()
    for ver in all_versions:
        if ver in MXNetGraph.registry_:
            ops.update(MXNetGraph.registry_[ver].keys())
    ops = list(ops)
    ops.sort()
    return ops