import logging
import unittest
from oslo_utils import importutils
import sqlalchemy as sa
from sqlalchemy import orm
from oslo_db import exception as db_exc
from oslo_db.sqlalchemy import models
from oslo_db import tests
from oslo_db.tests.sqlalchemy import base as test_base
def operate_on_row(name, ready=None, proceed=None):
    logging.debug('%s starting', name)
    _session = self.sessionmaker()
    with _session.begin():
        logging.debug('%s ready', name)
        tbl = self.test_table()
        tbl.update({'foo': 10})
        tbl.save(_session)
        if ready is not None:
            ready.send()
        if proceed is not None:
            logging.debug('%s waiting to proceed', name)
            proceed.wait()
        logging.debug('%s exiting transaction', name)
    logging.debug('%s terminating', name)
    return True