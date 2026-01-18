import sqlite3
from typing import (
@property
def rowcount(self) -> int:
    return self._cursor.rowcount