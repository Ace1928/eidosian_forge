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
def format_precompiler_header_cmd(compiler, head_file, head_file_pch, common_cflags, torch_include_dirs, extra_cflags, extra_include_paths):
    return re.sub('[ \\n]+', ' ', f'\n                {compiler} -x c++-header {head_file} -o {head_file_pch} {torch_include_dirs} {extra_include_paths} {extra_cflags} {common_cflags}\n            ').strip()