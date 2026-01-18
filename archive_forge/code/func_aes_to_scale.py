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
def aes_to_scale(var: str):
    """
    Look up the scale that should be used for a given aesthetic
    """
    if var in {'x', 'xmin', 'xmax', 'xend', 'xintercept'}:
        var = 'x'
    elif var in {'y', 'ymin', 'ymax', 'yend', 'yintercept'}:
        var = 'y'
    return var