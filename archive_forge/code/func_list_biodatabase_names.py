import os
from . import BioSeq
from . import Loader
from . import DBUtils
def list_biodatabase_names(self):
    """Return a list of all of the sub-databases."""
    return self.execute_and_fetch_col0('SELECT name FROM biodatabase')