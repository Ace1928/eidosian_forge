import datetime
import difflib
import functools
import inspect
import json
import os
import re
import tempfile
import threading
import unittest
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch._dynamo
import torch.utils._pytree as pytree
from torch._dynamo.utils import clone_input
from torch._subclasses.schema_check_mode import SchemaCheckMode
from torch._utils_internal import get_file_path_2
from torch.overrides import TorchFunctionMode
from torch.testing._internal.optests import (
def deepcopy_tensors(inputs: Any) -> Any:
    return pytree.tree_map_only(torch.Tensor, clone_input, inputs)