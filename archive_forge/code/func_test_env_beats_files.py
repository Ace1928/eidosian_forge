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
def test_env_beats_files(self):
    file_value = 'hello'
    env_value = 'goodbye'
    self.conf(args=[])
    self.conf_fixture.load_raw_values(group='foo', bar=file_value)
    self.assertEqual(file_value, self.conf['foo']['bar'])
    self.conf.reload_config_files()
    os.environ['OS_FOO__BAR'] = env_value
    self.assertEqual(env_value, self.conf['foo']['bar'])