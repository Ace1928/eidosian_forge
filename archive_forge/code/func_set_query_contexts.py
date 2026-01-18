from __future__ import annotations
import collections
import datetime
import functools
import glob
import itertools
import os
import random
import socket
import sqlite3
import string
import sys
import textwrap
import threading
import zlib
from typing import (
from coverage.debug import NoDebugging, auto_repr
from coverage.exceptions import CoverageException, DataError
from coverage.files import PathAliases
from coverage.misc import file_be_gone, isolate_module
from coverage.numbits import numbits_to_nums, numbits_union, nums_to_numbits
from coverage.sqlitedb import SqliteDb
from coverage.types import AnyCallable, FilePath, TArc, TDebugCtl, TLineNo, TWarnFn
from coverage.version import __version__
def set_query_contexts(self, contexts: Sequence[str] | None) -> None:
    """Set a number of contexts for subsequent querying.

        The next :meth:`lines`, :meth:`arcs`, or :meth:`contexts_by_lineno`
        calls will be limited to the specified contexts.  `contexts` is a list
        of Python regular expressions.  Contexts will be matched using
        :func:`re.search <python:re.search>`.  Data will be included in query
        results if they are part of any of the contexts matched.

        .. versionadded:: 5.0

        """
    self._start_using()
    if contexts:
        with self._connect() as con:
            context_clause = ' or '.join(['context regexp ?'] * len(contexts))
            with con.execute('select id from context where ' + context_clause, contexts) as cur:
                self._query_context_ids = [row[0] for row in cur.fetchall()]
    else:
        self._query_context_ids = None