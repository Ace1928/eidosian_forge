import logging
import unittest
from oslo_utils import importutils
import sqlalchemy as sa
from sqlalchemy import orm
from oslo_db import exception as db_exc
from oslo_db.sqlalchemy import models
from oslo_db import tests
from oslo_db.tests.sqlalchemy import base as test_base
@unittest.skipIf(not tests.should_run_eventlet_tests(), 'eventlet tests disabled unless TEST_EVENTLET=1')
def test_concurrent_transaction(self):
    sqla_logger = logging.getLogger('sqlalchemy.engine')
    sqla_logger.setLevel(logging.INFO)
    self.addCleanup(sqla_logger.setLevel, logging.NOTSET)

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
    eventlet = importutils.try_import('eventlet')
    if eventlet is None:
        return self.skipTest('eventlet is required for this test')
    a_ready = eventlet.event.Event()
    a_proceed = eventlet.event.Event()
    b_proceed = eventlet.event.Event()
    logging.debug('spawning A')
    a = eventlet.spawn(operate_on_row, 'A', ready=a_ready, proceed=a_proceed)
    logging.debug('waiting for A to enter transaction')
    a_ready.wait()
    logging.debug('spawning B')
    b = eventlet.spawn(operate_on_row, 'B', proceed=b_proceed)
    logging.debug('waiting for B to (attempt to) enter transaction')
    eventlet.sleep(1)
    a_proceed.send()
    self.assertTrue(a.wait())
    b_proceed.send()
    self.assertRaises(db_exc.DBDuplicateEntry, b.wait)