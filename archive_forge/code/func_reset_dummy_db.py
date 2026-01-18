import random
import string
import uuid
import warnings
import fixtures
from oslo_config import cfg
from oslo_db import options
from oslo_serialization import jsonutils
import sqlalchemy
from sqlalchemy import exc as sqla_exc
from heat.common import context
from heat.db import api as db_api
from heat.db import models
from heat.engine import environment
from heat.engine import node_data
from heat.engine import resource
from heat.engine import stack
from heat.engine import template
def reset_dummy_db():
    engine = get_engine()
    meta = sqlalchemy.MetaData()
    meta.reflect(bind=engine)
    for table in reversed(meta.sorted_tables):
        if table.name == 'migrate_version':
            continue
        with engine.connect() as conn, conn.begin():
            conn.execute(table.delete())