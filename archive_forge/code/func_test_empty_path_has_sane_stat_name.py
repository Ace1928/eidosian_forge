from unittest import mock
import uuid
from oslotest import base as test_base
import statsd
import webob.dec
import webob.exc
from oslo_middleware import stats
def test_empty_path_has_sane_stat_name(self):
    app = self.make_stats_middleware()
    path = '/'
    self.perform_request(app, path, 'GET')
    expected_stat = '{name}.{method}'.format(name=app.stat_name, method='GET')
    app.statsd.timer.assert_called_once_with(expected_stat)