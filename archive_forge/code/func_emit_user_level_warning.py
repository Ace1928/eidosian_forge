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
def emit_user_level_warning(message, category=None) -> None:
    """Emit a warning at the user level by inspecting the stack trace."""
    stacklevel = find_stack_level()
    return warnings.warn(message, category=category, stacklevel=stacklevel)