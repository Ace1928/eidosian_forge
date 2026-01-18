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
def is_calculated_aes(ae: Any) -> bool:
    """
    Return True if Aesthetic expression maps to calculated statistic

    This function is now only used to identify the deprecated versions
    e.g. "..var.." or "stat(var)".

    Parameters
    ----------
    ae : object
        Single aesthetic mapping

    >>> is_calculated_aes("density")
    False

    >>> is_calculated_aes(4)
    False

    >>> is_calculated_aes("..density..")
    True

    >>> is_calculated_aes("stat(density)")
    True

    >>> is_calculated_aes("stat(100*density)")
    True

    >>> is_calculated_aes("100*stat(density)")
    True
    """
    if not isinstance(ae, str):
        return False
    return any((pattern.search(ae) for pattern in (STAT_RE, DOTS_RE)))