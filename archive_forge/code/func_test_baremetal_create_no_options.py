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
def test_baremetal_create_no_options(self):
    arglist = []
    verifylist = []
    self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)