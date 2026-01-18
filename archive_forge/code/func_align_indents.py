import codecs
import functools
import importlib
import inspect
import json
import os
import re
import sys
import types
import warnings
from pathlib import Path
from textwrap import dedent, indent
from typing import (
import numpy as np
import pandas
from packaging import version
from pandas._typing import JSONSerializable
from pandas.util._decorators import Appender  # type: ignore
from pandas.util._print_versions import (  # type: ignore[attr-defined]
from modin._version import get_versions
from modin.config import DocModule, Engine, StorageFormat
def align_indents(source: str, target: str) -> str:
    """
    Align indents of two strings.

    Parameters
    ----------
    source : str
        Source string to align indents with.
    target : str
        Target string to align indents.

    Returns
    -------
    str
        Target string with indents aligned with the source.
    """
    source_indent = _get_indent(source)
    target = dedent(target)
    return indent(target, ' ' * source_indent)