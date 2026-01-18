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
def load_test_case(dir: str) -> Tuple[bytes, Any, Any]:
    """Load a self contained ONNX test case from a directory.

    The test case must contain the model and the inputs/outputs data. The directory structure
    should be as follows:

    dir
    ├── test_<name>
    │   ├── model.onnx
    │   └── test_data_set_0
    │       ├── input_0.pb
    │       ├── input_1.pb
    │       ├── output_0.pb
    │       └── output_1.pb

    Args:
        dir: The directory containing the test case.

    Returns:
        model_bytes: The ONNX model in bytes.
        inputs: the inputs data, mapping from input name to numpy.ndarray.
        outputs: the outputs data, mapping from output name to numpy.ndarray.
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:
        raise ImportError('Load test case from ONNX format failed: Please install ONNX.') from exc
    with open(os.path.join(dir, 'model.onnx'), 'rb') as f:
        model_bytes = f.read()
    test_data_dir = os.path.join(dir, 'test_data_set_0')
    inputs = {}
    input_files = glob.glob(os.path.join(test_data_dir, 'input_*.pb'))
    for input_file in input_files:
        tensor = onnx.load_tensor(input_file)
        inputs[tensor.name] = numpy_helper.to_array(tensor)
    outputs = {}
    output_files = glob.glob(os.path.join(test_data_dir, 'output_*.pb'))
    for output_file in output_files:
        tensor = onnx.load_tensor(output_file)
        outputs[tensor.name] = numpy_helper.to_array(tensor)
    return (model_bytes, inputs, outputs)