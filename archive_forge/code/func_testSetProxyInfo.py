from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import system_util
from gslib.utils import ls_helper
from gslib.utils import retry_util
from gslib.utils import text_util
from gslib.utils import unit_util
import gslib.tests.testcase as testcase
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import TestParams
from gslib.utils.text_util import CompareVersions
from gslib.utils.unit_util import DecimalShort
from gslib.utils.unit_util import HumanReadableWithDecimalPlaces
from gslib.utils.unit_util import PrettyTime
import httplib2
import os
import six
from six import add_move, MovedModule
from six.moves import mock
def testSetProxyInfo(self):
    """Tests SetProxyInfo for various proxy use cases in boto file."""
    valid_proxy_types = ['socks4', 'socks5', 'http']
    valid_proxy_host = ['hostname', '1.2.3.4', None]
    valid_proxy_port = [8888, 0]
    valid_proxy_user = ['foo', None]
    valid_proxy_pass = ['Bar', None]
    valid_proxy_rdns = [True, False, None]
    proxy_type_spec = {'socks4': httplib2.socks.PROXY_TYPE_SOCKS4, 'socks5': httplib2.socks.PROXY_TYPE_SOCKS5, 'http': httplib2.socks.PROXY_TYPE_HTTP, 'https': httplib2.socks.PROXY_TYPE_HTTP}
    boto_proxy_config_test_values = [{'proxy_host': p_h, 'proxy_type': p_t, 'proxy_port': p_p, 'proxy_user': p_u, 'proxy_pass': p_s, 'proxy_rdns': p_d} for p_h in valid_proxy_host for p_s in valid_proxy_pass for p_p in valid_proxy_port for p_u in valid_proxy_user for p_t in valid_proxy_types for p_d in valid_proxy_rdns]
    with SetEnvironmentForTest({'http_proxy': 'http://host:50'}):
        for test_values in boto_proxy_config_test_values:
            proxy_type = proxy_type_spec.get(test_values.get('proxy_type'))
            proxy_host = test_values.get('proxy_host')
            proxy_port = test_values.get('proxy_port')
            proxy_user = test_values.get('proxy_user')
            proxy_pass = test_values.get('proxy_pass')
            proxy_rdns = bool(test_values.get('proxy_rdns'))
            if not proxy_type == proxy_type_spec['http']:
                proxy_rdns = False
            expected = httplib2.ProxyInfo(proxy_host=proxy_host, proxy_type=proxy_type, proxy_port=proxy_port, proxy_user=proxy_user, proxy_pass=proxy_pass, proxy_rdns=proxy_rdns)
            if not (expected.proxy_host and expected.proxy_port):
                expected = httplib2.ProxyInfo(proxy_type_spec['http'], 'host', 50)
                if test_values.get('proxy_rdns') == None:
                    expected.proxy_rdns = True
            self._AssertProxyInfosEqual(boto_util.SetProxyInfo(test_values), expected)