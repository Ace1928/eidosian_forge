import argparse
import contextlib
import copy
import ctypes
import errno
import functools
import gc
import inspect
import io
import json
import logging
import math
import operator
import os
import platform
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import warnings
from collections.abc import Mapping, Sequence
from contextlib import closing, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from itertools import product, chain
from pathlib import Path
from statistics import mean
from typing import (
from unittest.mock import MagicMock
import expecttest
import numpy as np
import __main__  # type: ignore[import]
import torch
import torch.backends.cudnn
import torch.backends.mkl
import torch.backends.mps
import torch.backends.xnnpack
import torch.cuda
from torch import Tensor
from torch._C import ScriptDict, ScriptList  # type: ignore[attr-defined]
from torch._utils_internal import get_writable_path
from torch.nn import (
from torch.onnx import (
from torch.testing import make_tensor
from torch.testing._comparison import (
from torch.testing._comparison import not_close_error_metas
from torch.testing._internal.common_dtype import get_all_dtypes
import torch.utils._pytree as pytree
from .composite_compliance import no_dispatch
def check_nondeterministic_alert(self, fn, caller_name, should_alert=True):
    """Checks that an operation produces a nondeterministic alert when
        expected while `torch.use_deterministic_algorithms(True)` is set.

        Args:
          fn (callable): Function to check for a nondeterministic alert

          caller_name (str): Name of the operation that produces the
              nondeterministic alert. This name is expected to appear at the
              beginning of the error/warning message.

          should_alert (bool, optional): If True, then the check will only pass
              if calling `fn` produces a nondeterministic error/warning with the
              expected message. If False, then the check will only pass if
              calling `fn` does not produce an error. Default: `True`.
        """
    alert_message = '^' + caller_name + ' does not have a deterministic implementation, but you set'
    with DeterministicGuard(True):
        if should_alert:
            with self.assertRaisesRegex(RuntimeError, alert_message, msg='expected a non-deterministic error, but it was not raised'):
                fn()
        else:
            try:
                fn()
            except RuntimeError as e:
                if 'does not have a deterministic implementation' in str(e):
                    self.fail('did not expect non-deterministic error message, ' + 'but got one anyway: "' + str(e) + '"')
                raise
    with DeterministicGuard(True, warn_only=True):
        if should_alert:
            with self.assertWarnsRegex(UserWarning, alert_message):
                fn()
        else:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                fn()
                for warning in w:
                    if isinstance(warning, UserWarning):
                        self.assertTrue(re.search(alert_message, str(warning)) is None)