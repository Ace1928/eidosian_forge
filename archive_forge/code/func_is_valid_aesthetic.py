from __future__ import annotations
import re
import typing
from collections.abc import Iterable, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import fields
from typing import Any, Dict
import pandas as pd
from ..iapi import labels_view
from .evaluation import after_stat, stage
def is_valid_aesthetic(value: Any, ae: str) -> bool:
    """
    Return True if `value` looks valid.

    Parameters
    ----------
    value :
        Value to check
    ae :
        Aesthetic name

    Notes
    -----
    There are no guarantees that he value is spot on
    valid.
    """
    if ae == 'linetype':
        named = {'solid', 'dashed', 'dashdot', 'dotted', '_', '--', '-.', ':', 'none', ' ', ''}
        if value in named:
            return True
        conditions = [isinstance(value, tuple), isinstance(value[0], int), isinstance(value[1], tuple), len(value[1]) % 2 == 0, all((isinstance(x, int) for x in value[1]))]
        if all(conditions):
            return True
        return False
    elif ae == 'shape':
        if isinstance(value, str):
            return True
        conditions = [isinstance(value, tuple), all((isinstance(x, int) for x in value)), 0 <= value[1] < 3]
        if all(conditions):
            return True
        return False
    elif ae in {'color', 'fill'}:
        if isinstance(value, str):
            return True
        with suppress(TypeError):
            if isinstance(value, (tuple, list)) and all((0 <= x <= 1 for x in value)):
                return True
        return False
    return False