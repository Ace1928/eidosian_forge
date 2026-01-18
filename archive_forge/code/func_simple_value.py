import base64
import cloudpickle
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from triad import assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path
from tune._utils import product
from tune._utils.math import (
@property
def simple_value(self) -> Dict[str, Any]:
    """If the template contains no tuning expression, it's simple
        and it will return parameters dictionary, otherwise, ``ValueError``
        will be raised
        """
    assert_or_throw(self.empty, ValueError('template contains tuning expressions'))
    if len(self._func_positions) == 0:
        return self._template
    return self._fill_funcs(deepcopy(self._template))