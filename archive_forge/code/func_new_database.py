import os
from . import BioSeq
from . import Loader
from . import DBUtils
def new_database(self, db_name, authority=None, description=None):
    """Add a new database to the server and return it."""
    sql = 'INSERT INTO biodatabase (name, authority, description) VALUES (%s, %s, %s)'
    self.adaptor.execute(sql, (db_name, authority, description))
    return BioSeqDatabase(self.adaptor, db_name)