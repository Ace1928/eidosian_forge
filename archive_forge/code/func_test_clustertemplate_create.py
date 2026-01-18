import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import cluster_templates
def test_clustertemplate_create(self):
    cluster_template = self.mgr.create(**CREATE_CLUSTERTEMPLATE)
    expect = [('POST', '/v1/clustertemplates', {}, CREATE_CLUSTERTEMPLATE)]
    self.assertEqual(expect, self.api.calls)
    self.assertTrue(cluster_template)
    self.assertEqual(CLUSTERTEMPLATE1['docker_volume_size'], cluster_template.docker_volume_size)
    self.assertEqual(CLUSTERTEMPLATE1['docker_storage_driver'], cluster_template.docker_storage_driver)