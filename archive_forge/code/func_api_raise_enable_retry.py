from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
@api.safe_for_db_retry
def api_raise_enable_retry(self, *args, **kwargs):
    return self._api_raise(*args, **kwargs)