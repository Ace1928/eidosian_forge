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
def has_groups(data: pd.DataFrame) -> bool:
    """
    Check if data is grouped

    Parameters
    ----------
    data :
        Data

    Returns
    -------
    out : bool
        If True, the data has groups.
    """
    return data.loc[0, 'group'] != NO_GROUP