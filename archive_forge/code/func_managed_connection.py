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
@contextmanager
def managed_connection(self, commit_frequency=5000):
    try:
        con = self.connection or self._connect()
        self._initialize(con)
        yield con
    except ValueError as exc:
        if self.connection is None:
            con.close()
        raise exc
    else:
        if self.connection is None:
            con.commit()
            con.close()
        else:
            self.change_count += 1
            if self.change_count % commit_frequency == 0:
                con.commit()