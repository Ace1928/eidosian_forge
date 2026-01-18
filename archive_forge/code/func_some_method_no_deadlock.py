from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
@api.wrap_db_retry(max_retries=1, retry_on_deadlock=True)
def some_method_no_deadlock():
    raise exception.RetryRequest(ValueError())