import copy
import json
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_deploy_template
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_deploy_template_unset_no_property(self):
    uuid = baremetal_fakes.baremetal_deploy_template_uuid
    arglist = [uuid]
    verifylist = [('template', uuid)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.assertFalse(self.baremetal_mock.deploy_template.update.called)