import collections
import logging
import re
import sys
from sqlalchemy import event
from sqlalchemy import exc as sqla_exc
from oslo_db import exception
from oslo_db.sqlalchemy import compat
@event.listens_for(engine, 'rollback_savepoint')
def rollback_savepoint(conn, name, context):
    exc_info = sys.exc_info()
    if exc_info[1]:
        if not conn.invalidated:
            conn.info[ROLLBACK_CAUSE_KEY] = exc_info[1]
    del exc_info