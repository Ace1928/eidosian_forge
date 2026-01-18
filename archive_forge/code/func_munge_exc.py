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
def munge_exc(e, *, suppress_suffix=True, suppress_prefix=True, file=None, skip=0):
    if file is None:
        file = inspect.stack()[1 + skip].filename
    s = str(e)

    def repl_frame(m):
        if m.group(1) != file:
            return ''
        if m.group(2) == '<module>':
            return ''
        return m.group(0)
    s = re.sub('  File "([^"]+)", line \\d+, in (.+)\\n    .+\\n( +[~^]+ *\\n)?', repl_frame, s)
    s = re.sub('line \\d+', 'line N', s)
    s = re.sub(file, os.path.basename(file), s)
    s = re.sub(os.path.join(os.path.dirname(torch.__file__), ''), '', s)
    s = re.sub('\\\\', '/', s)
    if suppress_suffix:
        s = re.sub('\\n*Set TORCH_LOGS.+', '', s, flags=re.DOTALL)
        s = re.sub('\\n*You can suppress this exception.+', '', s, flags=re.DOTALL)
    if suppress_prefix:
        s = re.sub('Cannot export model.+\\n\\n', '', s)
    s = re.sub(' +$', '', s, flags=re.M)
    return s