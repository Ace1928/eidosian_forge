import atexit
import os
import re
import shutil
import textwrap
import threading
from typing import Any, List, Optional
import torch
from torch.utils.benchmark.utils._stubs import CallgrindModuleType, TimeitModuleType
from torch.utils.benchmark.utils.common import _make_temp_dir
from torch.utils import cpp_extension
def compile_timeit_template(*, stmt: str, setup: str, global_setup: str) -> TimeitModuleType:
    template_path: str = os.path.join(SOURCE_ROOT, 'timeit_template.cpp')
    with open(template_path) as f:
        src: str = f.read()
    module = _compile_template(stmt=stmt, setup=setup, global_setup=global_setup, src=src, is_standalone=False)
    assert isinstance(module, TimeitModuleType)
    return module