import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import cluster_templates
def test_clustertemplate_delete_by_name(self):
    cluster_template = self.mgr.delete(CLUSTERTEMPLATE1['name'])
    expect = [('DELETE', '/v1/clustertemplates/%s' % CLUSTERTEMPLATE1['name'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertIsNone(cluster_template)