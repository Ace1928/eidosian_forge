import datetime
from unittest import mock
from oslo_config import cfg
from oslo_utils import timeutils
from heat.common import context
from heat.common import service_utils
from heat.engine import service
from heat.engine import worker
from heat.objects import service as service_objects
from heat.rpc import worker_api
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
@mock.patch('oslo_log.log.setup')
def test_engine_service_reset(self, setup_logging_mock):
    self.eng.reset()
    setup_logging_mock.assert_called_once_with(cfg.CONF, 'heat')