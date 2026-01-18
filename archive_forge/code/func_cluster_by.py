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
def cluster_by(self, *expressions: t.Optional[ExpOrStr], append: bool=True, dialect: DialectType=None, copy: bool=True, **opts) -> Select:
    """
        Set the CLUSTER BY expression.

        Example:
            >>> Select().from_("tbl").select("x").cluster_by("x DESC").sql(dialect="hive")
            'SELECT x FROM tbl CLUSTER BY x DESC'

        Args:
            *expressions: the SQL code strings to parse.
                If a `Group` instance is passed, this is used as-is.
                If another `Expression` instance is passed, it will be wrapped in a `Cluster`.
            append: if `True`, add to any existing expressions.
                Otherwise, this flattens all the `Order` expression into a single expression.
            dialect: the dialect used to parse the input expression.
            copy: if `False`, modify this expression instance in-place.
            opts: other options to use to parse the input expressions.

        Returns:
            The modified Select expression.
        """
    return _apply_child_list_builder(*expressions, instance=self, arg='cluster', append=append, copy=copy, prefix='CLUSTER BY', into=Cluster, dialect=dialect, **opts)