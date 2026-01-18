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
def all_viable_backends(cls):
    """Return an iterator of all ``Backend`` objects that are present

        and provisionable.

        """
    for backend in cls.backends_by_database_type.values():
        try:
            yield backend._verify()
        except exception.BackendNotAvailable:
            pass