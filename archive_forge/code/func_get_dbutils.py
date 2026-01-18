import os
from typing import Dict, Type
def get_dbutils(module_name):
    """Return the correct dbutils object for the database driver."""
    try:
        return _dbutils[module_name]()
    except KeyError:
        return Generic_dbutils()