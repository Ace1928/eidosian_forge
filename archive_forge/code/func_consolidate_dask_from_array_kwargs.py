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
def consolidate_dask_from_array_kwargs(from_array_kwargs: dict[Any, Any], name: str | None=None, lock: bool | None=None, inline_array: bool | None=None) -> dict[Any, Any]:
    """
    Merge dask-specific kwargs with arbitrary from_array_kwargs dict.

    Temporary function, to be deleted once explicitly passing dask-specific kwargs to .chunk() is deprecated.
    """
    from_array_kwargs = _resolve_doubly_passed_kwarg(from_array_kwargs, kwarg_name='name', passed_kwarg_value=name, default=None, err_msg_dict_name='from_array_kwargs')
    from_array_kwargs = _resolve_doubly_passed_kwarg(from_array_kwargs, kwarg_name='lock', passed_kwarg_value=lock, default=False, err_msg_dict_name='from_array_kwargs')
    from_array_kwargs = _resolve_doubly_passed_kwarg(from_array_kwargs, kwarg_name='inline_array', passed_kwarg_value=inline_array, default=False, err_msg_dict_name='from_array_kwargs')
    return from_array_kwargs