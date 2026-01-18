from __future__ import annotations
import contextlib
import re
import sqlite3
from typing import cast, Any, Iterable, Iterator, Tuple
from coverage.debug import auto_repr, clipped_repr, exc_one_line
from coverage.exceptions import DataError
from coverage.types import TDebugCtl
def execute_one(self, sql: str, parameters: Iterable[Any]=()) -> tuple[Any, ...] | None:
    """Execute a statement and return the one row that results.

        This is like execute(sql, parameters).fetchone(), except it is
        correct in reading the entire result set.  This will raise an
        exception if more than one row results.

        Returns a row, or None if there were no rows.
        """
    with self.execute(sql, parameters) as cur:
        rows = list(cur)
    if len(rows) == 0:
        return None
    elif len(rows) == 1:
        return cast(Tuple[Any, ...], rows[0])
    else:
        raise AssertionError(f"SQL {sql!r} shouldn't return {len(rows)} rows")