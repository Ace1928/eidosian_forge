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
def test__make_client(self):
    conf = {'host': '127.0.0.1', 'port': 6379, 'username': 'default', 'password': 'secret', 'namespace': 'test'}
    test_conf = {'host': '127.0.0.1', 'port': 6379, 'username': 'default', 'password': 'secret'}
    with mock.patch('taskflow.utils.redis_utils.RedisClient') as mock_ru:
        impl_redis.RedisJobBoard('test-board', conf)
        mock_ru.assert_called_once_with(**test_conf)