import collections
import logging
import re
import sys
from sqlalchemy import event
from sqlalchemy import exc as sqla_exc
from oslo_db import exception
from oslo_db.sqlalchemy import compat
@event.listens_for(engine, 'checkin')
def pop_exc_checkin(dbapi_conn, connection_record):
    connection_record.info.pop(ROLLBACK_CAUSE_KEY, None)