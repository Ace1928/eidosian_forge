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
def instancer(_class: Callable[[], T]) -> T:
    """
    Create a dummy instance each time this is imported.

    This serves the purpose of allowing us to use all of pandas plotting methods
    without aliasing and writing each of them ourselves.

    Parameters
    ----------
    _class : object

    Returns
    -------
    object
        Instance of `_class`.
    """
    return _class()