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
def contexts_by_lineno(self, filename: str) -> dict[TLineNo, list[str]]:
    """Get the contexts for each line in a file.

        Returns:
            A dict mapping line numbers to a list of context names.

        .. versionadded:: 5.0

        """
    self._start_using()
    with self._connect() as con:
        file_id = self._file_id(filename)
        if file_id is None:
            return {}
        lineno_contexts_map = collections.defaultdict(set)
        if self.has_arcs():
            query = 'select arc.fromno, arc.tono, context.context ' + 'from arc, context ' + 'where arc.file_id = ? and arc.context_id = context.id'
            data = [file_id]
            if self._query_context_ids is not None:
                ids_array = ', '.join('?' * len(self._query_context_ids))
                query += ' and arc.context_id in (' + ids_array + ')'
                data += self._query_context_ids
            with con.execute(query, data) as cur:
                for fromno, tono, context in cur:
                    if fromno > 0:
                        lineno_contexts_map[fromno].add(context)
                    if tono > 0:
                        lineno_contexts_map[tono].add(context)
        else:
            query = 'select l.numbits, c.context from line_bits l, context c ' + 'where l.context_id = c.id ' + 'and file_id = ?'
            data = [file_id]
            if self._query_context_ids is not None:
                ids_array = ', '.join('?' * len(self._query_context_ids))
                query += ' and l.context_id in (' + ids_array + ')'
                data += self._query_context_ids
            with con.execute(query, data) as cur:
                for numbits, context in cur:
                    for lineno in numbits_to_nums(numbits):
                        lineno_contexts_map[lineno].add(context)
    return {lineno: list(contexts) for lineno, contexts in lineno_contexts_map.items()}