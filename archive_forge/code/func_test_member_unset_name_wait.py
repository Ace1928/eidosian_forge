import copy
from unittest import mock
import osc_lib.tests.utils as osc_test_utils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import member
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_member_unset_name_wait(self):
    self._test_member_unset_param_wait('name')