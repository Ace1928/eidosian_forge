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
def sanitize_special_tokens(self) -> int:
    """
        The `sanitize_special_tokens` is now deprecated kept for backward compatibility and will be removed in
        transformers v5.
        """
    logger.warning_once('The `sanitize_special_tokens` will be removed in transformers v5.')
    return self.add_tokens(self.all_special_tokens_extended, special_tokens=True)