import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.deploy_template
def test_deploy_templates_list(self):
    deploy_templates = self.mgr.list()
    expect = [('GET', '/v1/deploy_templates', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(1, len(deploy_templates))