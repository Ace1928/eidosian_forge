import copy
import json
import os
import warnings
from typing import Any, Dict, Optional, Union
from .. import __version__
from ..configuration_utils import PretrainedConfig
from ..utils import (
def dict_torch_dtype_to_str(self, d: Dict[str, Any]) -> None:
    """
        Checks whether the passed dictionary and its nested dicts have a *torch_dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
    if d.get('torch_dtype', None) is not None and (not isinstance(d['torch_dtype'], str)):
        d['torch_dtype'] = str(d['torch_dtype']).split('.')[1]
    for value in d.values():
        if isinstance(value, dict):
            self.dict_torch_dtype_to_str(value)