import functools
import os
import fixtures
from oslo_db import options as db_options
from oslo_db.sqlalchemy import enginefacade
from keystone.common import sql
import keystone.conf
from keystone.tests import unit
def recreate(self):
    sql.ModelBase.metadata.create_all(bind=self.engine)