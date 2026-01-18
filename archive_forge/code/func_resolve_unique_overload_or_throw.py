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
def resolve_unique_overload_or_throw(op: torch._ops.OpOverloadPacket) -> torch._ops.OpOverload:
    all_schemas = torch._C._jit_get_schemas_for_operator(op._qualified_op_name)
    if len(all_schemas) != 1:
        raise RuntimeError(f'opcheck can only test operators without overloads. Got the following overloads for {op._qualified_op_name}: {[schema.overload_name for schema in all_schemas]}')
    overload_name = all_schemas[0].overload_name
    if overload_name == '':
        return op.default
    return getattr(op, overload_name)