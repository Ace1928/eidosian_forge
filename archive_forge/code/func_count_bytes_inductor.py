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
@register_backend
def count_bytes_inductor(gm, example_inputs):
    return compile_fx(gm, example_inputs, inner_compile=count_bytes_inner)