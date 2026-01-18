import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.deploy_template
def test_deploy_templates_list_sort_dir(self):
    self.api = utils.FakeAPI(fake_responses_sorting)
    self.mgr = ironicclient.v1.deploy_template.DeployTemplateManager(self.api)
    deploy_templates = self.mgr.list(sort_dir='desc')
    expect = [('GET', '/v1/deploy_templates/?sort_dir=desc', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(2, len(deploy_templates))