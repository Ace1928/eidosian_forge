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
def get_atom_details(self, ad_uuid):
    try:
        atomdetails = self._tables.atomdetails
        with self._engine.begin() as conn:
            q = sql.select(atomdetails).where(atomdetails.c.uuid == ad_uuid)
            row = conn.execute(q).first()
            if not row:
                raise exc.NotFound("No atom details found with uuid '%s'" % ad_uuid)
            row = row._mapping
            return self._converter.convert_atom_detail(row)
    except sa_exc.SQLAlchemyError:
        exc.raise_with_cause(exc.StorageFailure, "Failed getting atom details with uuid '%s'" % ad_uuid)