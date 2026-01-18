from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
def api_raise_default(self, *args, **kwargs):
    return self._api_raise(*args, **kwargs)