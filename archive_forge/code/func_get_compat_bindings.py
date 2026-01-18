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
def get_compat_bindings() -> CallgrindModuleType:
    with LOCK:
        global COMPAT_CALLGRIND_BINDINGS
        if COMPAT_CALLGRIND_BINDINGS is None:
            COMPAT_CALLGRIND_BINDINGS = cpp_extension.load(name='callgrind_bindings', sources=[os.path.join(SOURCE_ROOT, 'valgrind_wrapper', 'compat_bindings.cpp')], extra_cflags=CXX_FLAGS, extra_include_paths=EXTRA_INCLUDE_PATHS)
    return COMPAT_CALLGRIND_BINDINGS