from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def from_db_cursor(cursor, **kwargs):
    if cursor.description:
        table = PrettyTable(**kwargs)
        table.field_names = [col[0] for col in cursor.description]
        for row in cursor.fetchall():
            table.add_row(row)
        return table