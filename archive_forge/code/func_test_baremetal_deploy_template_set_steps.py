import copy
import json
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_deploy_template
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_deploy_template_set_steps(self):
    arglist = [baremetal_fakes.baremetal_deploy_template_uuid, '--steps', baremetal_fakes.baremetal_deploy_template_steps]
    verifylist = [('template', baremetal_fakes.baremetal_deploy_template_uuid), ('steps', baremetal_fakes.baremetal_deploy_template_steps)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    expected_steps = json.loads(baremetal_fakes.baremetal_deploy_template_steps)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.deploy_template.update.assert_called_once_with(baremetal_fakes.baremetal_deploy_template_uuid, [{'path': '/steps', 'value': expected_steps, 'op': 'add'}])