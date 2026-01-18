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
def convert_added_tokens(cls, obj: Union[AddedToken, Any], save=False, add_type_field=True):
    if isinstance(obj, dict) and '__type' in obj and (obj['__type'] == 'AddedToken'):
        obj.pop('__type')
        return AddedToken(**obj)
    if isinstance(obj, AddedToken) and save:
        obj = obj.__getstate__()
        if add_type_field:
            obj['__type'] = 'AddedToken'
        else:
            obj.pop('special')
        return obj
    elif isinstance(obj, (list, tuple)):
        return [cls.convert_added_tokens(o, save=save, add_type_field=add_type_field) for o in obj]
    elif isinstance(obj, dict):
        return {k: cls.convert_added_tokens(v, save=save, add_type_field=add_type_field) for k, v in obj.items()}
    return obj