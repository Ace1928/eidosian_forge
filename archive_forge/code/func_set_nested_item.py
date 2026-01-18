import functools
import logging
import operator
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy
import wasabi.tables
from .. import util
from ..errors import Errors
from ..pipeline import MultiLabel_TextCategorizer, TextCategorizer
from ..training import Corpus
from ._util import Arg, Opt, app, import_code, setup_gpu
def set_nested_item(config: Dict[str, Any], keys: List[str], value: float) -> Dict[str, Any]:
    """Set item in nested dictionary. Adapted from https://stackoverflow.com/a/54138200.
        config (Dict[str, Any]): Configuration dictionary.
        keys (List[Any]): Path to value to set.
        value (float): Value to set.
        RETURNS (Dict[str, Any]): Updated dictionary.
        """
    functools.reduce(operator.getitem, keys[:-1], config)[keys[-1]] = value
    return config