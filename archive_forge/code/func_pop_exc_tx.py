import collections
import logging
import re
import sys
from sqlalchemy import event
from sqlalchemy import exc as sqla_exc
from oslo_db import exception
from oslo_db.sqlalchemy import compat
@event.listens_for(engine, 'rollback')
@event.listens_for(engine, 'commit')
def pop_exc_tx(conn):
    if not conn.invalidated:
        conn.info.pop(ROLLBACK_CAUSE_KEY, None)