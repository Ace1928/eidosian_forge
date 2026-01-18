import unittest
from functools import partial
from typing import List
import numpy as np
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import SM53OrLater
from torch.testing._internal.common_device_type import precisionOverride
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import TEST_SCIPY, TEST_WITH_ROCM
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.opinfo.refs import (
def error_inputs_fft(op_info, device, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(make_arg()), error_type=IndexError, error_regex='Dimension specified as -1 but tensor has no dimensions')