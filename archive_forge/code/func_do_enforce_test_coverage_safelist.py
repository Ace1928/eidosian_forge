import itertools
import os
import platform
import unittest
from typing import Any, Optional, Sequence, Tuple
import numpy
import onnx.backend.base
import onnx.backend.test
import onnx.shape_inference
import onnx.version_converter
from onnx import ModelProto, NodeProto, TensorProto
from onnx.backend.base import Device, DeviceType
from onnx.backend.test.runner import BackendIsNotSupposedToImplementIt
def do_enforce_test_coverage_safelist(model: ModelProto) -> bool:
    if model.graph.name not in test_coverage_safelist:
        return False
    return all((node.op_type not in {'RNN', 'LSTM', 'GRU'} for node in model.graph.node))