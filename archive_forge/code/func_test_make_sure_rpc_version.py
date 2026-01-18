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
def test_make_sure_rpc_version(self):
    self.assertEqual('1.36', service.EngineService.RPC_API_VERSION, 'RPC version is changed, please update this test to new version and make sure additional test cases are added for RPC APIs added in new version')