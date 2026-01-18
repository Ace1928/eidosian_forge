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
@mock.patch('heat.common.messaging.get_rpc_server', return_value=mock.Mock())
@mock.patch('oslo_messaging.Target', return_value=mock.Mock())
@mock.patch('heat.common.messaging.get_rpc_client', return_value=mock.Mock())
@mock.patch('heat.common.service_utils.generate_engine_id', return_value=mock.Mock())
@mock.patch('heat.engine.service.ThreadGroupManager', return_value=mock.Mock())
@mock.patch('heat.engine.service.EngineListener', return_value=mock.Mock())
@mock.patch('heat.engine.worker.WorkerService', return_value=mock.Mock())
@mock.patch('oslo_service.threadgroup.ThreadGroup', return_value=mock.Mock())
@mock.patch.object(service.EngineService, '_configure_db_conn_pool_size')
def test_engine_service_start_in_convergence_mode(self, configure_db_conn_pool_size, thread_group_class, worker_service_class, engine_listener_class, thread_group_manager_class, sample_uuid_method, rpc_client_class, target_class, rpc_server_method):
    cfg.CONF.set_override('convergence_engine', True)
    self._test_engine_service_start(thread_group_class, worker_service_class, engine_listener_class, thread_group_manager_class, sample_uuid_method, rpc_client_class, target_class, rpc_server_method)