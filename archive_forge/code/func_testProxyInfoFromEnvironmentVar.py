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
def testProxyInfoFromEnvironmentVar(self):
    """Tests ProxyInfoFromEnvironmentVar for various cases."""
    valid_variables = ['http_proxy', 'https_proxy']
    if not system_util.IS_WINDOWS:
        valid_variables.append('HTTPS_PROXY')
    clear_dict = {}
    for key in valid_variables:
        clear_dict[key] = None
    with SetEnvironmentForTest(clear_dict):
        for env_var in valid_variables:
            for url_string in ['hostname', 'http://hostname', 'https://hostname']:
                with SetEnvironmentForTest({env_var: url_string}):
                    self._AssertProxyInfosEqual(boto_util.ProxyInfoFromEnvironmentVar(env_var), httplib2.ProxyInfo(httplib2.socks.PROXY_TYPE_HTTP, 'hostname', 443 if env_var.lower().startswith('https') else 80))
                    for other_env_var in valid_variables:
                        if other_env_var == env_var:
                            continue
                        self._AssertProxyInfosEqual(boto_util.ProxyInfoFromEnvironmentVar(other_env_var), httplib2.ProxyInfo(httplib2.socks.PROXY_TYPE_HTTP, None, 0))
            for url_string in ['1.2.3.4:50', 'http://1.2.3.4:50', 'https://1.2.3.4:50']:
                with SetEnvironmentForTest({env_var: url_string}):
                    self._AssertProxyInfosEqual(boto_util.ProxyInfoFromEnvironmentVar(env_var), httplib2.ProxyInfo(httplib2.socks.PROXY_TYPE_HTTP, '1.2.3.4', 50))
            for url_string in ['foo:bar@1.2.3.4:50', 'http://foo:bar@1.2.3.4:50', 'https://foo:bar@1.2.3.4:50']:
                with SetEnvironmentForTest({env_var: url_string}):
                    self._AssertProxyInfosEqual(boto_util.ProxyInfoFromEnvironmentVar(env_var), httplib2.ProxyInfo(httplib2.socks.PROXY_TYPE_HTTP, '1.2.3.4', 50, proxy_user='foo', proxy_pass='bar'))
            for url_string in ['bar@1.2.3.4:50', 'http://bar@1.2.3.4:50', 'https://bar@1.2.3.4:50']:
                with SetEnvironmentForTest({env_var: url_string}):
                    self._AssertProxyInfosEqual(boto_util.ProxyInfoFromEnvironmentVar(env_var), httplib2.ProxyInfo(httplib2.socks.PROXY_TYPE_HTTP, '1.2.3.4', 50, proxy_user='bar'))
        for env_var in ['proxy', 'noproxy', 'garbage']:
            for url_string in ['1.2.3.4:50', 'http://1.2.3.4:50']:
                with SetEnvironmentForTest({env_var: url_string}):
                    self._AssertProxyInfosEqual(boto_util.ProxyInfoFromEnvironmentVar(env_var), httplib2.ProxyInfo(httplib2.socks.PROXY_TYPE_HTTP, None, 0))