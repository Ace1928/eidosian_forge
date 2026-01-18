import os
from oslotest import base
from requests import HTTPError
import requests_mock
import testtools
from oslo_config import _list_opts
from oslo_config import cfg
from oslo_config import fixture
from oslo_config import sources
from oslo_config.sources import _uri
def test_invalid_env(self):
    self.conf(args=[])
    env_value = 'ABC'
    os.environ['OS_FOO__BAZ'] = env_value
    with testtools.ExpectedException(cfg.ConfigSourceValueError):
        self.conf['foo']['baz']