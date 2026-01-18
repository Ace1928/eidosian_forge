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
@classmethod
def register_for_auto_class(cls, auto_class='AutoTokenizer'):
    """
        Register this class with a given auto class. This should only be used for custom tokenizers as the ones in the
        library are already mapped with `AutoTokenizer`.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoTokenizer"`):
                The auto class to register this new tokenizer with.
        """
    if not isinstance(auto_class, str):
        auto_class = auto_class.__name__
    import transformers.models.auto as auto_module
    if not hasattr(auto_module, auto_class):
        raise ValueError(f'{auto_class} is not a valid auto class.')
    cls._auto_class = auto_class