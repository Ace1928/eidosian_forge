from __future__ import annotations
import abc
import pickle
import time
from typing import Any
from typing import Callable
from typing import cast
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Union
from ..util.typing import Self
@classmethod
def from_config_dict(cls, config_dict: Mapping[str, Any], prefix: str) -> Self:
    prefix_len = len(prefix)
    return cls(dict(((key[prefix_len:], config_dict[key]) for key in config_dict if key.startswith(prefix))))