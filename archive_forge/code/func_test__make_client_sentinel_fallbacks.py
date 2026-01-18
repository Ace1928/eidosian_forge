import time
from unittest import mock
from oslo_utils import uuidutils
import testtools
from taskflow import exceptions as excp
from taskflow.jobs.backends import impl_redis
from taskflow import states
from taskflow import test
from taskflow.tests.unit.jobs import base
from taskflow.tests import utils as test_utils
from taskflow.utils import persistence_utils as p_utils
from taskflow.utils import redis_utils as ru
def test__make_client_sentinel_fallbacks(self):
    conf = {'host': '127.0.0.1', 'port': 26379, 'username': 'default', 'password': 'secret', 'namespace': 'test', 'sentinel': 'mymaster', 'sentinel_fallbacks': ['[::1]:26379', '127.0.0.2:26379', 'localhost:26379']}
    with mock.patch('redis.sentinel.Sentinel') as mock_sentinel:
        impl_redis.RedisJobBoard('test-board', conf)
        test_conf = {'username': 'default', 'password': 'secret', 'sentinel_kwargs': None}
        mock_sentinel.assert_called_once_with([('127.0.0.1', 26379), ('::1', 26379), ('127.0.0.2', 26379), ('localhost', 26379)], **test_conf)
        mock_sentinel().master_for.assert_called_once_with('mymaster')