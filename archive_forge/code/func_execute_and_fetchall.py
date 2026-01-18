import os
from . import BioSeq
from . import Loader
from . import DBUtils
def execute_and_fetchall(self, sql, args=None):
    """Return a list of tuples of all rows."""
    out = super().execute_and_fetchall(sql, args)
    return [tuple((self._bytearray_to_str(v) for v in o)) for o in out]