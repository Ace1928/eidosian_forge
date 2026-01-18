import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import cluster_templates
def test_clustertemplate_delete_by_id(self):
    cluster_template = self.mgr.delete(CLUSTERTEMPLATE1['id'])
    expect = [('DELETE', '/v1/clustertemplates/%s' % CLUSTERTEMPLATE1['id'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertIsNone(cluster_template)