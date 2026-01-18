from oslo_db.sqlalchemy import models
from oslo_utils import uuidutils
import sqlalchemy as sa
from sqlalchemy.ext import declarative
from sqlalchemy import orm
from neutron_lib.db import constants as db_const
def get_tenant_id(self):
    return self.project_id