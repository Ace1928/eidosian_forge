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
def generate_repro(test: str, op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any], *, save_data: bool, dry_run: bool=False) -> str:
    if save_data:
        now = datetime.datetime.now()
        path = os.path.join(tempfile.gettempdir(), 'pytorch_opcheck_safe_to_delete')
        unix_timestamp = datetime.datetime.timestamp(now) * 100000
        filepath = os.path.join(path, f'repro_{unix_timestamp}.pt')
        if not dry_run:
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save((args, kwargs), filepath)
        args_kwargs = f'args, kwargs = torch.load("{filepath}")'
    else:
        args_kwargs = '# If you rerun your test with PYTORCH_OPCHECK_PRINT_BETTER_REPRO=1\n# we will fill them in same (args, kwargs) as in your test\nargs = ()  # args to the operator\nkwargs = {}  # kwargs to the operator'
    ns, name = op._schema.name.split('::')
    overload = op._overloadname
    repro_command = f'# =========================================================\n# BEGIN REPRO SCRIPT\n# =========================================================\nimport torch\nfrom torch.testing._internal.optests import opcheck\n\n# Make sure you have loaded the library that contains the op\n# via an import or torch.ops.load_library(...)\nop = torch.ops.{ns}.{name}.{overload}\n\n{args_kwargs}\nopcheck(op, args, kwargs, test_utils="{test}")\n# =========================================================\n# END REPRO SCRIPT\n# =========================================================\n'
    return repro_command