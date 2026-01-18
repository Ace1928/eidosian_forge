import argparse
import itertools
import os
from typing import Sequence, TypeVar, Union
from libfb.py.log import set_simple_logging  # type: ignore[import]
from torchgen import gen
from torchgen.context import native_function_manager
from torchgen.model import DispatchKey, NativeFunctionsGroup, NativeFunctionsViewGroup
from torchgen.static_runtime import config, generator
def clang_format(cpp_file_path: str) -> None:
    import subprocess
    subprocess.check_call(['clang-format', '-i', cpp_file_path])