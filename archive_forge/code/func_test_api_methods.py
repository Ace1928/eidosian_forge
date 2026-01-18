import eventlet  # noqa
import functools   # noqa: E402
import inspect   # noqa: E402
import os   # noqa: E402
from unittest import mock   # noqa: E402
import fixtures   # noqa: E402
from oslo_concurrency import lockutils   # noqa: E402
from oslo_config import cfg   # noqa: E402
from oslo_config import fixture as config_fixture   # noqa: E402
from oslo_log.fixture import logging_error   # noqa: E402
import testtools   # noqa: E402
from oslo_versionedobjects.tests import obj_fixtures   # noqa: E402
def test_api_methods(self):
    self.assertTrue(self.cover_api is not None)
    api_methods = [x for x in dir(self.cover_api) if not x.startswith('_')]
    test_methods = [x[5:] for x in dir(self) if x.startswith('test_')]
    self.assertThat(test_methods, testtools.matchers.ContainsAll(api_methods))