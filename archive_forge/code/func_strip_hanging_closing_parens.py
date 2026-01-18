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
def strip_hanging_closing_parens(s):
    """
        Remove leftover  parens
        """
    stack = 0
    idx = []
    for i, c in enumerate(s):
        if c == '(':
            stack += 1
        elif c == ')':
            stack -= 1
            if stack < 0:
                idx.append(i)
                stack = 0
                continue
        yield c