import collections
from unittest import mock
from barbicanclient import exceptions
from heat.common import exception
from heat.engine.clients.os import barbican
from heat.tests import common
from heat.tests import utils
def test_get_secret_payload_by_ref(self):
    payload_content = 'payload content'
    secret = collections.namedtuple('Secret', ['name', 'payload'])('foo', payload_content)
    self.barbican_client.secrets.get.return_value = secret
    expect = payload_content
    self.assertEqual(expect, self.barbican_plugin.get_secret_payload_by_ref('secret'))