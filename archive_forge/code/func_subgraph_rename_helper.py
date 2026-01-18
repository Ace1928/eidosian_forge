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
def subgraph_rename_helper(name: str) -> Any:
    if name in sg_rename:
        return sg_rename[name]
    return rename_helper(name)