import json
import numbers
import os
import sqlite3
import sys
from contextlib import contextmanager
import numpy as np
import ase.io.jsonio
from ase.data import atomic_numbers
from ase.calculators.calculator import all_properties
from ase.db.row import AtomsRow
from ase.db.core import (Database, ops, now, lock, invop, parse_selection,
from ase.parallel import parallel_function
def delete_external_table(self, name):
    """Delete an external table."""
    if not self._external_table_exists(name):
        return
    with self.managed_connection() as con:
        cur = con.cursor()
        sql = 'DROP TABLE {}'.format(name)
        cur.execute(sql)
        sql = 'DELETE FROM information WHERE value=?'
        cur.execute(sql, (name,))
        sql = 'DELETE FROM information WHERE name=?'
        cur.execute(sql, (name + '_dtype',))