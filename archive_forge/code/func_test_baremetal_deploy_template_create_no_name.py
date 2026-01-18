import copy
import json
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_deploy_template
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_deploy_template_create_no_name(self):
    arglist = ['--steps', baremetal_fakes.baremetal_deploy_template_steps]
    verifylist = [('steps', baremetal_fakes.baremetal_deploy_template_steps)]
    self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)
    self.assertFalse(self.baremetal_mock.deploy_template.create.called)