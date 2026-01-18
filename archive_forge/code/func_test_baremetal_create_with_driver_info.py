import copy
import io
import json
import sys
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient.common import utils as commonutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_node
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
from ironicclient.v1 import utils as v1_utils
def test_baremetal_create_with_driver_info(self):
    self.check_with_options(['--driver-info', 'arg1=val1', '--driver-info', 'arg2=val2'], [('driver_info', ['arg1=val1', 'arg2=val2'])], {'driver_info': {'arg1': 'val1', 'arg2': 'val2'}})