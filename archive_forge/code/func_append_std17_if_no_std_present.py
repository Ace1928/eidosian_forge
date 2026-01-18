import copy
import glob
import importlib
import importlib.abc
import os
import re
import shlex
import shutil
import setuptools
import subprocess
import sys
import sysconfig
import warnings
import collections
from pathlib import Path
import errno
import torch
import torch._appdirs
from .file_baton import FileBaton
from ._cpp_extension_versioner import ExtensionVersioner
from .hipify import hipify_python
from .hipify.hipify_python import GeneratedFileCleaner
from typing import Dict, List, Optional, Union, Tuple
from torch.torch_version import TorchVersion, Version
from setuptools.command.build_ext import build_ext
def append_std17_if_no_std_present(cflags) -> None:
    cpp_format_prefix = '/{}:' if self.compiler.compiler_type == 'msvc' else '-{}='
    cpp_flag_prefix = cpp_format_prefix.format('std')
    cpp_flag = cpp_flag_prefix + 'c++17'
    if not any((flag.startswith(cpp_flag_prefix) for flag in cflags)):
        cflags.append(cpp_flag)