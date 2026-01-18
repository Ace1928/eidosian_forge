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
def create_named_database(self, engine, ident, conditional=False):
    with engine.connect().execution_options(isolation_level='AUTOCOMMIT') as conn:
        if not conditional or not self.database_exists(conn, ident):
            conn.exec_driver_sql('CREATE DATABASE %s' % ident)