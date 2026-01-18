import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import onnx
from onnx.backend.test.case.test_case import TestCase
from onnx.backend.test.case.utils import import_recursive
from onnx.onnx_pb import (
def rename_helper(internal_name: str) -> Any:
    if internal_name in io_names_map:
        return io_names_map[internal_name]
    elif internal_name == '':
        return ''
    return op_prefix + internal_name