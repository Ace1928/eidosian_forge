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
def get_atoms_for_flow(self, fd_uuid):
    gathered = []
    try:
        with self._engine.connect() as conn:
            for ad in self._converter.atom_query_iter(conn, fd_uuid):
                gathered.append(ad)
    except sa_exc.DBAPIError:
        exc.raise_with_cause(exc.StorageFailure, "Failed getting atom details in flow detail '%s'" % fd_uuid)
    for atom_details in gathered:
        yield atom_details