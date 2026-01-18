import json
from unittest import mock
import ddt
from osprofiler.drivers import loginsight
from osprofiler import exc
from osprofiler.tests import test
@mock.patch.object(loginsight.LogInsightClient, '_is_current_session_active', return_value=True)
@mock.patch.object(loginsight.LogInsightClient, '_send_request')
def test_login_with_current_session_active(self, send_request, is_current_session_active):
    self._client.login()
    is_current_session_active.assert_called_once_with()
    send_request.assert_not_called()