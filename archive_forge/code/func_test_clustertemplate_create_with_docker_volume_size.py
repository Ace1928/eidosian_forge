import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import cluster_templates
def test_clustertemplate_create_with_docker_volume_size(self):
    cluster_template_with_docker_volume_size = dict()
    cluster_template_with_docker_volume_size.update(CREATE_CLUSTERTEMPLATE)
    cluster_template_with_docker_volume_size['docker_volume_size'] = 11
    cluster_template = self.mgr.create(**cluster_template_with_docker_volume_size)
    expect = [('POST', '/v1/clustertemplates', {}, cluster_template_with_docker_volume_size)]
    self.assertEqual(expect, self.api.calls)
    self.assertTrue(cluster_template)
    self.assertEqual(CLUSTERTEMPLATE1['docker_volume_size'], cluster_template.docker_volume_size)
    self.assertEqual(CLUSTERTEMPLATE1['docker_storage_driver'], cluster_template.docker_storage_driver)