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
@mock.patch('heat.common.context.get_admin_context', return_value=mock.Mock())
@mock.patch('heat.objects.service.Service.delete', return_value=mock.Mock())
def test_engine_service_stop_in_non_convergence_mode(self, service_delete_method, admin_context_method):
    cfg.CONF.set_default('convergence_engine', False)
    self._test_engine_service_stop(service_delete_method, admin_context_method)