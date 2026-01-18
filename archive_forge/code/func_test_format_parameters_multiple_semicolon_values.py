import os
from unittest import mock
from urllib import request
import testtools
from heatclient.common import utils
from heatclient import exc
from heatclient.v1 import resources as hc_res
def test_format_parameters_multiple_semicolon_values(self):
    p = utils.format_parameters(['KeyName=heat_key', 'DnsSecKey=hsgx1m31;PbaNF4WEcHlwj;IlCGgfOdoB;58/ww7a4oAO;NQ/fD==', 'UpstreamDNS=8.8.8.8'])
    self.assertEqual({'KeyName': 'heat_key', 'DnsSecKey': 'hsgx1m31;PbaNF4WEcHlwj;IlCGgfOdoB;58/ww7a4oAO;NQ/fD==', 'UpstreamDNS': '8.8.8.8'}, p)