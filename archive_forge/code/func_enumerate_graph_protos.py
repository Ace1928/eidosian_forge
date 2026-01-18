from __future__ import annotations
import os
import sys
from typing import Any, Iterable
import numpy as np
import onnx
import onnx.external_data_helper as ext_data
import onnx.helper
import onnx.onnx_cpp2py_export.checker as c_checker
def enumerate_graph_protos(self) -> Iterable[onnx.GraphProto]:
    """Enumerates all GraphProtos in a model."""
    yield self.model_proto.graph
    yield from _enumerate_subgraphs(self.model_proto.graph)