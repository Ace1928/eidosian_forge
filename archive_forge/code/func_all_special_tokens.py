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
@property
def all_special_tokens(self) -> List[str]:
    """
        `List[str]`: A list of the unique special tokens (`'<unk>'`, `'<cls>'`, ..., etc.).

        Convert tokens of `tokenizers.AddedToken` type to string.
        """
    all_toks = [str(s) for s in self.all_special_tokens_extended]
    return all_toks