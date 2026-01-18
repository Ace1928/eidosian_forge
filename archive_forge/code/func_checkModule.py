from torch.autograd import Variable
from torch.autograd.function import _nested_map
from torch.jit.annotations import BroadcastingList2, BroadcastingList3  # noqa: F401
from torch.onnx import OperatorExportTypes
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
import torch.jit.quantized
import zipfile
import functools
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_WINDOWS, \
from torch.testing._internal.common_jit import JitCommonTestCase
from torch.testing._internal.common_utils import enable_profiling_mode  # noqa: F401
from contextlib import contextmanager
from functools import reduce
from io import StringIO
from collections import defaultdict
import importlib.util
import inspect
import io
import math
import os
import pickle
import sys
import tempfile
import textwrap
from importlib.abc import Loader
from typing import Any, Dict, List, Tuple, Union
def checkModule(self, nn_module, args):
    """
        Check that a nn.Module's results in Script mode match eager and that it
        can be exported
        """
    sm = torch.jit.script(nn_module)
    with freeze_rng_state():
        eager_out = nn_module(*args)
    with freeze_rng_state():
        script_out = sm(*args)
    self.assertEqual(eager_out, script_out)
    self.assertExportImportModule(sm, args)
    return sm