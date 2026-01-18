from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from .._utils import add_margins, cross_join, join_keys, match, ninteraction
from ..exceptions import PlotnineError
from .facet import (
from .strips import Strips, strip
def ensure_list_spec(term: Sequence[str] | str) -> Sequence[str]:
    """
    Convert a str specification to a list spec

    e.g.
    'a' -> ['a']
    'a + b' -> ['a', 'b']
    '.' -> []
    '' -> []
    """
    if isinstance(term, str):
        splitter = ' + ' if ' + ' in term else '+'
        if term in ['.', '']:
            return []
        return [var.strip() for var in term.split(splitter)]
    else:
        return term