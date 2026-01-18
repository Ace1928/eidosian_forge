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
def check_compiler_is_gcc(compiler):
    if not IS_LINUX:
        return False
    env = os.environ.copy()
    env['LC_ALL'] = 'C'
    version_string = subprocess.check_output([compiler, '-v'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
    pattern = re.compile('^COLLECT_GCC=(.*)$', re.MULTILINE)
    results = re.findall(pattern, version_string)
    if len(results) != 1:
        return False
    compiler_path = os.path.realpath(results[0].strip())
    if os.path.basename(compiler_path) == 'c++' and 'gcc version' in version_string:
        return True
    return False