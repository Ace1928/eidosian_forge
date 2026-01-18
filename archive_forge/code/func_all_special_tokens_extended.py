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
def all_special_tokens_extended(self) -> List[Union[str, AddedToken]]:
    """
        `List[Union[str, tokenizers.AddedToken]]`: All the special tokens (`'<unk>'`, `'<cls>'`, etc.), the order has
        nothing to do with the index of each tokens. If you want to know the correct indices, check
        `self.added_tokens_encoder`. We can't create an order anymore as the keys are `AddedTokens` and not `Strings`.

        Don't convert tokens of `tokenizers.AddedToken` type to string so they can be used to control more finely how
        special tokens are tokenized.
        """
    all_tokens = []
    seen = set()
    for value in self.special_tokens_map_extended.values():
        if isinstance(value, (list, tuple)):
            tokens_to_add = [token for token in value if str(token) not in seen]
        else:
            tokens_to_add = [value] if str(value) not in seen else []
        seen.update(map(str, tokens_to_add))
        all_tokens.extend(tokens_to_add)
    return all_tokens