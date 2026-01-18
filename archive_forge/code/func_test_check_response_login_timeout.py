import json
from unittest import mock
import ddt
from osprofiler.drivers import loginsight
from osprofiler import exc
from osprofiler.tests import test
def test_check_response_login_timeout(self):
    resp = mock.Mock(status_code=440)
    self.assertRaises(exc.LogInsightLoginTimeout, self._client._check_response, resp)