import os
from unittest import mock
from urllib import request
import testtools
from heatclient.common import utils
from heatclient import exc
from heatclient.v1 import resources as hc_res
def test_format_parameters_parse_semicolon_false(self):
    p = utils.format_parameters(['KeyName=heat_key;UpstreamDNS=8.8.8.8;a=b'], parse_semicolon=False)
    self.assertEqual({'KeyName': 'heat_key;UpstreamDNS=8.8.8.8;a=b'}, p)