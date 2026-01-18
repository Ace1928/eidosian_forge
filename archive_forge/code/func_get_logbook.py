import copy
import functools
import threading
import time
from oslo_utils import strutils
import sqlalchemy as sa
from sqlalchemy import exc as sa_exc
from sqlalchemy import pool as sa_pool
from sqlalchemy import sql
import tenacity
from taskflow import exceptions as exc
from taskflow import logging
from taskflow.persistence.backends.sqlalchemy import migration
from taskflow.persistence.backends.sqlalchemy import tables
from taskflow.persistence import base
from taskflow.persistence import models
from taskflow.utils import eventlet_utils
from taskflow.utils import misc
def get_logbook(self, book_uuid, lazy=False):
    try:
        logbooks = self._tables.logbooks
        with self._engine.connect() as conn:
            q = sql.select(logbooks).where(logbooks.c.uuid == book_uuid)
            row = conn.execute(q).first()
            if not row:
                raise exc.NotFound("No logbook found with uuid '%s'" % book_uuid)
            row = row._mapping
            book = self._converter.convert_book(row)
            if not lazy:
                self._converter.populate_book(conn, book)
            return book
    except sa_exc.DBAPIError:
        exc.raise_with_cause(exc.StorageFailure, "Failed getting logbook '%s'" % book_uuid)