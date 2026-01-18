from __future__ import annotations
import base64
import copyreg
import dataclasses
import functools
import hashlib
import importlib
import io
import json
import logging
import multiprocessing
import os
import pathlib
import pickle
import pkgutil
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import threading
import warnings
import weakref
from bisect import bisect_right
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from copy import copy
from ctypes import c_void_p, cdll, CDLL
from dataclasses import field
from functools import partial
from importlib import abc
from pathlib import Path
from threading import Thread
from time import sleep, time
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union
import torch
from torch._dynamo.device_interface import (
from torch._dynamo.utils import counters
from torch._inductor import config, exc
from torch._inductor.codegen.cuda import cuda_env
from torch._inductor.utils import cache_dir, developer_warning, is_linux
from torch._prims_common import suggest_memory_format
from torch.fx.experimental.symbolic_shapes import has_hint, hint_int, ShapeEnv
from torch.hub import _Faketqdm, tqdm
import torch
from ctypes import cdll
def extract_tensor_metadata(t: torch.Tensor) -> TensorMetadata:
    """
    Extract the TensorMetadata of a tensor.
    """
    memory_format: Optional[torch.memory_format] = suggest_memory_format(t)
    if not t.is_contiguous(memory_format=memory_format):
        memory_format = None
    return TensorMetadata(dtype=t.dtype, shape=t.shape, stride=t.stride() if t.layout == torch.strided else (), device=t.device, layout=t.layout, memory_format=memory_format, storage_offset=t.storage_offset(), requires_grad=t.requires_grad, is_quantized=t.is_quantized, is_conj=t.is_conj(), is_neg=t.is_neg(), is_coalesced=t.is_coalesced() if t.is_sparse else False, dense_dim=t.dense_dim() if t.is_sparse else False, sparse_dim=t.sparse_dim() if t.is_sparse else False)