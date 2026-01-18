from .db_utilities import decode_torsion, decode_matrices, db_hash
from .sage_helper import _within_sage
from spherogram.codecs import DTcodec
import sys
import sqlite3
import re
import random
import importlib
import collections
def mfld_hash(manifold):
    """
    We cache the hash to speed up searching for one manifold in
    multiple tables.
    """
    if 'db_hash' not in manifold._cache:
        manifold._cache['db_hash'] = db_hash(manifold)
    return manifold._cache['db_hash']