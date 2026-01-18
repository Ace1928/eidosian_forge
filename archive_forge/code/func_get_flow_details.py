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
def get_flow_details(self, fd_uuid, lazy=False):
    try:
        flowdetails = self._tables.flowdetails
        with self._engine.begin() as conn:
            q = sql.select(flowdetails).where(flowdetails.c.uuid == fd_uuid)
            row = conn.execute(q).first()
            if not row:
                raise exc.NotFound("No flow details found with uuid '%s'" % fd_uuid)
            row = row._mapping
            fd = self._converter.convert_flow_detail(row)
            if not lazy:
                self._converter.populate_flow_detail(conn, fd)
            return fd
    except sa_exc.SQLAlchemyError:
        exc.raise_with_cause(exc.StorageFailure, "Failed getting flow details with uuid '%s'" % fd_uuid)