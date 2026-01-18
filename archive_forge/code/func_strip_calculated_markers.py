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
def strip_calculated_markers(value):
    """
    Remove markers for calculated aesthetics

    Parameters
    ----------
    value : object
        Aesthetic value. In most cases this will be a string
        but other types will pass through unmodified.

    Return
    ------
    out : object
        Aesthetic value with the dots removed.
    """
    return strip_stat(strip_dots(value))