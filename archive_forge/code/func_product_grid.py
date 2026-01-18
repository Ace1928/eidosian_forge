import base64
import cloudpickle
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from triad import assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path
from tune._utils import product
from tune._utils.math import (
def product_grid(self) -> Iterable['TuningParametersTemplate']:
    """cross product all grid parameters

        :yield: new templates with the grid paramters filled

        .. code-block:: python

            assert [dict(a=1,b=Rand(0,1)), dict(a=2,b=Rand(0,1))] ==                 list(to_template(dict(a=Grid(1,2),b=Rand(0,1))).product_grid())
        """
    if not self.has_grid:
        yield self
    else:
        gu: List[Tuple[int, List[Any]]] = [(i, list(u.expr)) for i, u in enumerate(self._units) if isinstance(u.expr, Grid)]
        yield from self._partial_fill([x[0] for x in gu], product([data for _, data in gu], safe=True, remove_empty=True))