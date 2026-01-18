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
@max_len_sentences_pair.setter
def max_len_sentences_pair(self, value) -> int:
    if value == self.model_max_length - self.num_special_tokens_to_add(pair=True) and self.verbose:
        if not self.deprecation_warnings.get('max_len_sentences_pair', False):
            logger.warning("Setting 'max_len_sentences_pair' is now deprecated. This value is automatically set up.")
        self.deprecation_warnings['max_len_sentences_pair'] = True
    else:
        raise ValueError("Setting 'max_len_sentences_pair' is now deprecated. This value is automatically set up.")