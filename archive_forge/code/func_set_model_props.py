import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def set_model_props(model: ModelProto, dict_value: Dict[str, str]) -> None:
    set_metadata_props(model, dict_value)