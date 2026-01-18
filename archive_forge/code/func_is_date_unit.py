from __future__ import annotations
import datetime
import inspect
import logging
import re
import sys
import typing as t
from collections.abc import Collection, Set
from contextlib import contextmanager
from copy import copy
from enum import Enum
from itertools import count
def is_date_unit(expression: t.Optional[exp.Expression]) -> bool:
    return expression is not None and expression.name.lower() in DATE_UNITS