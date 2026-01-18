import os
from unittest import mock
from urllib import request
import testtools
from heatclient.common import utils
from heatclient import exc
from heatclient.v1 import resources as hc_res
def test_newline_list_formatter(self):
    self.assertEqual('', utils.newline_list_formatter(None))
    self.assertEqual('', utils.newline_list_formatter([]))
    self.assertEqual('one\ntwo', utils.newline_list_formatter(['one', 'two']))