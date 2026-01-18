import abc
import logging
import os
import random
import re
import string
import sqlalchemy
from sqlalchemy import schema
from sqlalchemy import sql
import testresources
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import utils
def drop_named_database(self, engine, ident, conditional=False):
    with engine.connect().execution_options(isolation_level='AUTOCOMMIT') as conn:
        self._close_out_database_users(conn, ident)
        if conditional:
            conn.exec_driver_sql('DROP DATABASE IF EXISTS %s' % ident)
        else:
            conn.exec_driver_sql('DROP DATABASE %s' % ident)