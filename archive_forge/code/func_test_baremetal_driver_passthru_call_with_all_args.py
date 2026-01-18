import copy
from osc_lib.tests import utils as oscutils
from ironicclient.osc.v1 import baremetal_driver
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_driver_passthru_call_with_all_args(self):
    arglist = [baremetal_fakes.baremetal_driver_name, baremetal_fakes.baremetal_driver_passthru_method, '--arg', 'arg1=val1', '--arg', 'arg2=val2', '--http-method', 'POST']
    verifylist = [('driver', baremetal_fakes.baremetal_driver_name), ('method', baremetal_fakes.baremetal_driver_passthru_method), ('arg', ['arg1=val1', 'arg2=val2']), ('http_method', 'POST')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    args = [baremetal_fakes.baremetal_driver_name, baremetal_fakes.baremetal_driver_passthru_method]
    kwargs = {'http_method': 'POST', 'args': {'arg1': 'val1', 'arg2': 'val2'}}
    self.baremetal_mock.driver.vendor_passthru.assert_called_once_with(*args, **kwargs)