import torch
import re
import unittest
from subprocess import CalledProcessError
from torch._inductor.codecache import CppCodeCache
from torch.utils._triton import has_triton
from torch.testing._internal.common_utils import (
from torch._dynamo.backends.registry import register_backend
from torch._inductor.compile_fx import compile_fx, count_bytes_inner
from torch.testing._internal.common_utils import TestCase
def test_cpu():
    try:
        CppCodeCache.load('')
        return not IS_FBCODE
    except (CalledProcessError, OSError, torch._inductor.exc.InvalidCxxCompiler, torch._inductor.exc.CppCompileError):
        return False