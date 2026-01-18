import numpy as np
import torch
from contextlib import contextmanager
from torch.testing._internal.common_utils import TEST_WITH_ASAN, TEST_WITH_TSAN, TEST_WITH_UBSAN, IS_PPC, IS_MACOS, IS_WINDOWS
def qengine_is_fbgemm():
    return torch.backends.quantized.engine == 'fbgemm'