from __future__ import annotations
import json
import sqlite3
from itertools import zip_longest
from typing import Iterable
def register_sqlite_functions(connection: sqlite3.Connection) -> None:
    """
    Define numbits functions in a SQLite connection.

    This defines these functions for use in SQLite statements:

    * :func:`numbits_union`
    * :func:`numbits_intersection`
    * :func:`numbits_any_intersection`
    * :func:`num_in_numbits`
    * :func:`numbits_to_nums`

    `connection` is a :class:`sqlite3.Connection <python:sqlite3.Connection>`
    object.  After creating the connection, pass it to this function to
    register the numbits functions.  Then you can use numbits functions in your
    queries::

        import sqlite3
        from coverage.numbits import register_sqlite_functions

        conn = sqlite3.connect("example.db")
        register_sqlite_functions(conn)
        c = conn.cursor()
        # Kind of a nonsense query:
        # Find all the files and contexts that executed line 47 in any file:
        c.execute(
            "select file_id, context_id from line_bits where num_in_numbits(?, numbits)",
            (47,)
        )
    """
    connection.create_function('numbits_union', 2, numbits_union)
    connection.create_function('numbits_intersection', 2, numbits_intersection)
    connection.create_function('numbits_any_intersection', 2, numbits_any_intersection)
    connection.create_function('num_in_numbits', 2, num_in_numbits)
    connection.create_function('numbits_to_nums', 1, lambda b: json.dumps(numbits_to_nums(b)))