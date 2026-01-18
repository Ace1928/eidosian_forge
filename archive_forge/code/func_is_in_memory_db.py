import multiprocessing
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from django.db import NotSupportedError
from django.db.backends.base.creation import BaseDatabaseCreation
@staticmethod
def is_in_memory_db(database_name):
    return not isinstance(database_name, Path) and (database_name == ':memory:' or 'mode=memory' in database_name)