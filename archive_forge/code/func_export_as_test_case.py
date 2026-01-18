from __future__ import annotations
import glob
import io
import os
import shutil
import zipfile
from typing import Any, List, Mapping, Set, Tuple, Union
import torch
import torch.jit._trace
import torch.serialization
from torch.onnx import _constants, _exporter_states, errors
from torch.onnx._internal import _beartype, jit_utils, registration
@_beartype.beartype
def export_as_test_case(model_bytes: bytes, inputs_data, outputs_data, name: str, dir: str) -> str:
    """Export an ONNX model as a self contained ONNX test case.

    The test case contains the model and the inputs/outputs data. The directory structure
    is as follows:

    dir
    ├── test_<name>
    │   ├── model.onnx
    │   └── test_data_set_0
    │       ├── input_0.pb
    │       ├── input_1.pb
    │       ├── output_0.pb
    │       └── output_1.pb

    Args:
        model_bytes: The ONNX model in bytes.
        inputs_data: The inputs data, nested data structure of numpy.ndarray.
        outputs_data: The outputs data, nested data structure of numpy.ndarray.

    Returns:
        The path to the test case directory.
    """
    try:
        import onnx
    except ImportError as exc:
        raise ImportError('Export test case to ONNX format failed: Please install ONNX.') from exc
    test_case_dir = os.path.join(dir, 'test_' + name)
    os.makedirs(test_case_dir, exist_ok=True)
    _export_file(model_bytes, os.path.join(test_case_dir, 'model.onnx'), _exporter_states.ExportTypes.PROTOBUF_FILE, {})
    data_set_dir = os.path.join(test_case_dir, 'test_data_set_0')
    if os.path.exists(data_set_dir):
        shutil.rmtree(data_set_dir)
    os.makedirs(data_set_dir)
    proto = onnx.load_model_from_string(model_bytes)
    for i, (input_proto, input) in enumerate(zip(proto.graph.input, inputs_data)):
        export_data(input, input_proto, os.path.join(data_set_dir, f'input_{i}.pb'))
    for i, (output_proto, output) in enumerate(zip(proto.graph.output, outputs_data)):
        export_data(output, output_proto, os.path.join(data_set_dir, f'output_{i}.pb'))
    return test_case_dir