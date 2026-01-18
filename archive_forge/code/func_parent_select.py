from __future__ import annotations
import datetime
import math
import numbers
import re
import textwrap
import typing as t
from collections import deque
from copy import deepcopy
from enum import auto
from functools import reduce
from sqlglot.errors import ErrorLevel, ParseError
from sqlglot.helper import (
from sqlglot.tokens import Token
@property
def parent_select(self) -> t.Optional[Select]:
    """
        Returns the parent select statement.
        """
    return self.find_ancestor(Select)