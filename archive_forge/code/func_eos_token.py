import copy
import json
import os
import re
import warnings
from collections import UserDict
from collections.abc import Mapping, Sized
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import numpy as np
from packaging import version
from . import __version__
from .dynamic_module_utils import custom_object_save
from .utils import (
@eos_token.setter
def eos_token(self, value):
    if not isinstance(value, (str, AddedToken)) and value is not None:
        raise ValueError('Cannot set a non-string value as the EOS token')
    self._eos_token = value