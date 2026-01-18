import copy
from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import senlin
from heat.engine.resources.openstack.senlin import cluster as sc
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
from openstack import exceptions
def test_cluster_create_error(self):
    cfg.CONF.set_override('action_retry_limit', 0)
    cluster = self._init_cluster(self.t)
    self.senlin_mock.create_cluster.return_value = self.fake_cl
    mock_cluster = mock.MagicMock()
    mock_cluster.status = 'ERROR'
    mock_cluster.status_reason = 'oops'
    self.senlin_mock.get_policy.return_value = mock.Mock(id='fake_policy_id')
    self.senlin_mock.get_cluster.return_value = mock_cluster
    create_task = scheduler.TaskRunner(cluster.create)
    ex = self.assertRaises(exception.ResourceFailure, create_task)
    expected = 'ResourceInError: resources.senlin-cluster: Went to status ERROR due to "oops"'
    self.assertEqual(expected, str(ex))