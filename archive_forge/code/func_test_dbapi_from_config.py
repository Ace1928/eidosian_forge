from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
def test_dbapi_from_config(self):
    conf = cfg.ConfigOpts()
    dbapi = api.DBAPI.from_config(conf, backend_mapping={'sqlalchemy': __name__})
    self.assertIsNotNone(dbapi._backend)