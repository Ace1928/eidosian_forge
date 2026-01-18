import abc
import logging
import os
import random
import re
import string
import sqlalchemy
from sqlalchemy import schema
from sqlalchemy import sql
import testresources
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import utils
@classmethod
def backend_for_database_type(cls, database_type):
    """Return the ``Backend`` for the given database type.

        """
    try:
        backend = cls.backends_by_database_type[database_type]
    except KeyError:
        raise exception.BackendNotAvailable("Backend '%s' is unavailable: No such backend" % database_type)
    else:
        return backend._verify()