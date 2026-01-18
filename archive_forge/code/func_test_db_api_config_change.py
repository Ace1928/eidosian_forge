import sys
from unittest import mock
from oslo_config import fixture as config_fixture
from oslo_db import concurrency
from oslo_db.tests import base as test_base
@mock.patch('oslo_db.api.DBAPI')
def test_db_api_config_change(self, mock_db_api):
    fake_db_api = mock.MagicMock()
    mock_db_api.from_config.return_value = fake_db_api
    self.conf.set_override('use_tpool', True, group='database')
    self.db_api.fake_call
    mock_db_api.from_config.assert_called_once_with(conf=self.conf, backend_mapping=FAKE_BACKEND_MAPPING)
    self.eventlet.tpool.Proxy.assert_called_once_with(fake_db_api)
    self.assertEqual(self.proxy, self.db_api._db_api)