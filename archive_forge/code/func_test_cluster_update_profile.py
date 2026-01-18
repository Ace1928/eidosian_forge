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
def test_cluster_update_profile(self):
    cluster = self._create_cluster(self.t)
    self.senlin_mock.get_profile.side_effect = [mock.Mock(id='new_profile_id'), mock.Mock(id='fake_profile_id'), mock.Mock(id='new_profile_id')]
    new_t = copy.deepcopy(self.t)
    props = new_t['resources']['senlin-cluster']['properties']
    props['profile'] = 'new_profile'
    props['name'] = 'new_name'
    rsrc_defns = template.Template(new_t).resource_definitions(self.stack)
    new_cluster = rsrc_defns['senlin-cluster']
    self.senlin_mock.update_cluster.return_value = mock.Mock(cluster=new_cluster)
    self.senlin_mock.get_action.return_value = mock.Mock(status='SUCCEEDED')
    scheduler.TaskRunner(cluster.update, new_cluster)()
    self.assertEqual((cluster.UPDATE, cluster.COMPLETE), cluster.state)
    cluster_update_kwargs = {'profile_id': 'new_profile_id', 'name': 'new_name'}
    self.senlin_mock.update_cluster.assert_called_once_with(cluster=self.fake_cl, **cluster_update_kwargs)
    self.assertEqual(1, self.senlin_mock.get_action.call_count)