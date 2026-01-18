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
def test_cli_beats_env(self):
    env_value = 'goodbye'
    cli_value = 'cli'
    os.environ['OS_FOO__BAR'] = env_value
    self.conf.register_cli_opt(cfg.StrOpt('bar'), 'foo')
    self.conf(args=['--foo=%s' % cli_value])
    self.assertEqual(cli_value, self.conf['foo']['bar'])