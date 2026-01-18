import inspect
import pathlib
import sys
import typing
from collections import defaultdict
from types import CodeType
from typing import Dict, Iterable, List, Optional
import torch
def jit_code_filter(code: CodeType) -> bool:
    """Codefilter for Torchscript to trace forward calls.

    The custom CodeFilter is required while scripting a FX Traced forward calls.
    FX Traced forward calls have `code.co_filename` start with '<' which is used
    to exclude tracing of stdlib and site-packages in the default code filter.
    Since we need all forward calls to be traced, this custom code filter
    checks for code.co_name to be 'forward' and enables tracing for all such calls.
    The code filter is similar to default code filter for monkeytype and
    excludes tracing of stdlib and site-packages.
    """
    if code.co_name != 'forward' and (not code.co_filename or code.co_filename[0] == '<'):
        return False
    filename = pathlib.Path(code.co_filename).resolve()
    return not any((_startswith(filename, lib_path) for lib_path in LIB_PATHS))