from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import neutron
from heat.engine.clients.os import sahara
from heat.engine.resources.openstack.sahara import cluster as sc
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_cluster_create_no_image_anywhere_fails(self):
    self.t['resources']['super-cluster']['properties'].pop('default_image_id')
    self.sahara_mock.cluster_templates.get.return_value = mock.Mock(default_image_id=None)
    cluster = self._init_cluster(self.t)
    ex = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(cluster.create))
    self.assertIsInstance(ex.exc, exception.StackValidationFailed)
    self.assertIn('default_image_id must be provided: Referenced cluster template some_cluster_template_id has no default_image_id defined.', str(ex.message))