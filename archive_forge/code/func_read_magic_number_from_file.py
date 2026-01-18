from __future__ import annotations
import contextlib
import functools
import inspect
import io
import itertools
import math
import os
import re
import sys
import warnings
from collections.abc import (
from enum import Enum
from pathlib import Path
from typing import (
import numpy as np
import pandas as pd
from xarray.namedarray.utils import (  # noqa: F401
def read_magic_number_from_file(filename_or_obj, count=8) -> bytes:
    if isinstance(filename_or_obj, bytes):
        magic_number = filename_or_obj[:count]
    elif isinstance(filename_or_obj, io.IOBase):
        if filename_or_obj.tell() != 0:
            filename_or_obj.seek(0)
        magic_number = filename_or_obj.read(count)
        filename_or_obj.seek(0)
    else:
        raise TypeError(f'cannot read the magic number from {type(filename_or_obj)}')
    return magic_number