import os
import time
import math
import itertools
import numpy as np
from scipy.spatial.distance import cdist
from ase.io import write, read
from ase.geometry.cell import cell_to_cellpar
from ase.data import covalent_radii
from ase.ga import get_neighbor_list
def db_call_with_error_tol(db_cursor, expression, args=[]):
    """In case the GA is used on older versions of networking
    filesystems there might be some delays. For this reason
    some extra error tolerance when calling the SQLite db is
    employed.
    """
    import sqlite3
    i = 0
    while i < 10:
        try:
            db_cursor.execute(expression, args)
            return
        except sqlite3.OperationalError as e:
            print(e)
            time.sleep(2.0)
        i += 1
    raise sqlite3.OperationalError('Database still locked after 10 attempts (20 s)')