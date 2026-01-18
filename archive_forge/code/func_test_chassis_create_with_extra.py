import copy
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_chassis
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_chassis_create_with_extra(self):
    extra1 = 'arg1=val1'
    extra2 = 'arg2=val2'
    self.check_with_options(['--extra', extra1, '--extra', extra2], [('extra', [extra1, extra2])], {'extra': {'arg1': 'val1', 'arg2': 'val2'}})