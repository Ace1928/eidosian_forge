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
def install_gcc_via_conda() -> str:
    """On older systems, this is a quick way to get a modern compiler"""
    prefix = os.path.join(cache_dir(), 'gcc')
    cxx_path = os.path.join(prefix, 'bin', 'g++')
    if not os.path.exists(cxx_path):
        log.info('Downloading GCC via conda')
        conda = os.environ.get('CONDA_EXE', 'conda')
        if conda is None:
            conda = shutil.which('conda')
        if conda is not None:
            subprocess.check_call([conda, 'create', f'--prefix={prefix}', '--channel=conda-forge', '--quiet', '-y', 'python=3.8', 'gxx'], stdout=subprocess.PIPE)
    return cxx_path