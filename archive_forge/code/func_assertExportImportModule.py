import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
import torch.jit.quantized
from torch.testing._internal.common_dtype import floating_and_complex_types_and
from torch.testing._internal.common_utils import TestCase, \
from torch.testing._internal.common_utils import enable_profiling_mode  # noqa: F401
from itertools import chain
from typing import List, Union
from torch._C import TensorType
import io
def assertExportImportModule(self, m, inputs):
    m_import = self.getExportImportCopy(m)
    a = self.runAndSaveRNG(m, inputs)
    b = self.runAndSaveRNG(m_import, inputs)
    self.assertEqual(a, b, 'Results of original model and exported/imported version of model differed')