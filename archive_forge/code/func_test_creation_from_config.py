import logging
import os
from unittest import mock
import fixtures
from oslo_config import cfg
import sqlalchemy
from sqlalchemy.engine import base as base_engine
from sqlalchemy import exc
from sqlalchemy.pool import NullPool
from sqlalchemy import sql
from sqlalchemy import Column, MetaData, Table
from sqlalchemy import Integer, String
from sqlalchemy.orm import declarative_base
from oslo_db import exception
from oslo_db import options as db_options
from oslo_db.sqlalchemy import compat
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import models
from oslo_db.sqlalchemy import session
from oslo_db.sqlalchemy import utils
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
@mock.patch('oslo_db.sqlalchemy.orm.get_maker')
@mock.patch('oslo_db.sqlalchemy.engines.create_engine')
def test_creation_from_config(self, create_engine, get_maker):
    conf = cfg.ConfigOpts()
    conf.register_opts(db_options.database_opts, group='database')
    overrides = {'connection': 'sqlite:///:memory:', 'slave_connection': None, 'connection_debug': 100, 'max_pool_size': 10, 'mysql_sql_mode': 'TRADITIONAL'}
    for optname, optvalue in overrides.items():
        conf.set_override(optname, optvalue, group='database')
    session.EngineFacade.from_config(conf, expire_on_commit=True)
    create_engine.assert_called_once_with(sql_connection='sqlite:///:memory:', connection_debug=100, max_pool_size=10, mysql_sql_mode='TRADITIONAL', mysql_wsrep_sync_wait=None, sqlite_fk=False, connection_recycle_time=mock.ANY, retry_interval=mock.ANY, max_retries=mock.ANY, max_overflow=mock.ANY, connection_trace=mock.ANY, sqlite_synchronous=mock.ANY, pool_timeout=mock.ANY, thread_checkin=mock.ANY, json_serializer=None, json_deserializer=None, connection_parameters='', logging_name=mock.ANY)
    get_maker.assert_called_once_with(engine=create_engine(), expire_on_commit=True)