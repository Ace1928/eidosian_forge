from unittest import mock
from heatclient import exc
from heatclient.osc.v1 import template
from heatclient.tests.unit.osc.v1 import fakes
from heatclient.v1 import template_versions
def test_function_list_not_found(self):
    arglist = ['bad_version']
    self.template_versions.get.side_effect = exc.HTTPNotFound
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)