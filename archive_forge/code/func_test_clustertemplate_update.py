import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import cluster_templates
def test_clustertemplate_update(self):
    patch = {'op': 'replace', 'value': NEW_NAME, 'path': '/name'}
    cluster_template = self.mgr.update(id=CLUSTERTEMPLATE1['id'], patch=patch)
    expect = [('PATCH', '/v1/clustertemplates/%s' % CLUSTERTEMPLATE1['id'], {}, patch)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(NEW_NAME, cluster_template.name)