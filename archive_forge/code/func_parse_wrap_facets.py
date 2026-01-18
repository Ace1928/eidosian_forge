from __future__ import annotations
import re
import typing
from warnings import warn
import numpy as np
import pandas as pd
from .._utils import join_keys, match
from ..exceptions import PlotnineError, PlotnineWarning
from .facet import (
from .strips import Strips, strip
def parse_wrap_facets(facets: Optional[str | Sequence[str]]) -> Sequence[str]:
    """
    Return list of facetting variables
    """
    if facets is None:
        return []
    elif isinstance(facets, str):
        if '~' in facets:
            return parse_wrap_facets_old(facets)
        else:
            return [facets]
    return facets