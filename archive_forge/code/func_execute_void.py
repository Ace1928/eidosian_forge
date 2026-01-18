from __future__ import annotations
import contextlib
import re
import sqlite3
from typing import cast, Any, Iterable, Iterator, Tuple
from coverage.debug import auto_repr, clipped_repr, exc_one_line
from coverage.exceptions import DataError
from coverage.types import TDebugCtl
def execute_void(self, sql: str, parameters: Iterable[Any]=(), fail_ok: bool=False) -> None:
    """Same as :meth:`python:sqlite3.Connection.execute` when you don't need the cursor.

        If `fail_ok` is True, then SQLite errors are ignored.
        """
    try:
        self._execute(sql, parameters).close()
    except DataError:
        if not fail_ok:
            raise