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
def export_data(data, value_info_proto, f: str) -> None:
    """Export data to ONNX protobuf format.

    Args:
        data: The data to export, nested data structure of numpy.ndarray.
        value_info_proto: The ValueInfoProto of the data. The type of the ValueInfoProto
            determines how the data is stored.
        f: The file to write the data to.
    """
    try:
        from onnx import numpy_helper
    except ImportError as exc:
        raise ImportError('Export data to ONNX format failed: Please install ONNX.') from exc
    with open(f, 'wb') as opened_file:
        if value_info_proto.type.HasField('map_type'):
            opened_file.write(numpy_helper.from_dict(data, value_info_proto.name).SerializeToString())
        elif value_info_proto.type.HasField('sequence_type'):
            opened_file.write(numpy_helper.from_list(data, value_info_proto.name).SerializeToString())
        elif value_info_proto.type.HasField('optional_type'):
            opened_file.write(numpy_helper.from_optional(data, value_info_proto.name).SerializeToString())
        else:
            assert value_info_proto.type.HasField('tensor_type')
            opened_file.write(numpy_helper.from_array(data, value_info_proto.name).SerializeToString())