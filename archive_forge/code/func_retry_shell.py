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
def retry_shell(command, cwd=None, env=None, stdout=None, stderr=None, timeout=None, retries=1, was_rerun=False) -> Tuple[int, bool]:
    assert retries >= 0, f'Expecting non negative number for number of retries, got {retries}'
    try:
        exit_code = shell(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, timeout=timeout)
        if exit_code == 0 or retries == 0:
            return (exit_code, was_rerun)
        print(f'Got exit code {exit_code}, retrying (retries left={retries})', file=stdout, flush=True)
    except subprocess.TimeoutExpired:
        if retries == 0:
            print(f'Command took >{timeout // 60}min, returning 124', file=stdout, flush=True)
            return (124, was_rerun)
        print(f'Command took >{timeout // 60}min, retrying (retries left={retries})', file=stdout, flush=True)
    return retry_shell(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, timeout=timeout, retries=retries - 1, was_rerun=True)