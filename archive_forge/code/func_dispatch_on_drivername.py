import collections
from collections import abc
import itertools
import logging
import re
from oslo_utils import timeutils
import sqlalchemy
from sqlalchemy import Boolean
from sqlalchemy.engine import Connectable
from sqlalchemy.engine import url as sa_url
from sqlalchemy import exc
from sqlalchemy import func
from sqlalchemy import Index
from sqlalchemy import inspect
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy.sql.expression import cast
from sqlalchemy.sql.expression import literal_column
from sqlalchemy.sql import text
from sqlalchemy import Table
from oslo_db._i18n import _
from oslo_db import exception
from oslo_db.sqlalchemy import models
def dispatch_on_drivername(self, drivername):
    """Return a sub-dispatcher for the given drivername.

        This provides a means of calling a different function, such as the
        "*" function, for a given target object that normally refers
        to a sub-function.

        """
    dbname, driver = self._db_plus_driver_reg.match(drivername).group(1, 2)

    def go(*arg, **kw):
        return self._dispatch_on_db_driver(dbname, '*', arg, kw)
    return go