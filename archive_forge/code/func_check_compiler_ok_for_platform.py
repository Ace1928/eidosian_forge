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
def check_compiler_ok_for_platform(compiler: str) -> bool:
    """
    Verify that the compiler is the expected one for the current platform.

    Args:
        compiler (str): The compiler executable to check.

    Returns:
        True if the compiler is gcc/g++ on Linux or clang/clang++ on macOS,
        and always True for Windows.
    """
    if IS_WINDOWS:
        return True
    which = subprocess.check_output(['which', compiler], stderr=subprocess.STDOUT)
    compiler_path = os.path.realpath(which.decode(*SUBPROCESS_DECODE_ARGS).strip())
    if any((name in compiler_path for name in _accepted_compilers_for_platform())):
        return True
    env = os.environ.copy()
    env['LC_ALL'] = 'C'
    version_string = subprocess.check_output([compiler, '-v'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
    if IS_LINUX:
        pattern = re.compile('^COLLECT_GCC=(.*)$', re.MULTILINE)
        results = re.findall(pattern, version_string)
        if len(results) != 1:
            return 'clang version' in version_string
        compiler_path = os.path.realpath(results[0].strip())
        if os.path.basename(compiler_path) == 'c++' and 'gcc version' in version_string:
            return True
        return any((name in compiler_path for name in _accepted_compilers_for_platform()))
    if IS_MACOS:
        return version_string.startswith('Apple clang')
    return False