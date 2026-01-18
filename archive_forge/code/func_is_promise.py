import copy
import inspect
import io
import re
import warnings
from configparser import (
from dataclasses import dataclass
from pathlib import Path
from types import GeneratorType
from typing import (
import srsly
from .util import SimpleFrozenDict, SimpleFrozenList  # noqa: F401
@classmethod
def is_promise(cls, obj: Any) -> bool:
    """Check whether an object is a "promise", i.e. contains a reference
        to a registered function (via a key starting with `"@"`.
        """
    if not hasattr(obj, 'keys'):
        return False
    id_keys = [k for k in obj.keys() if k.startswith('@')]
    if len(id_keys):
        return True
    return False