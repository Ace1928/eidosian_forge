import io
from requests_mock.contrib import fixture
import testtools
from barbicanclient import barbican as barb
from barbicanclient.barbican import Barbican
from barbicanclient import client
from barbicanclient import exceptions
from barbicanclient.tests import keystone_client_fixtures
def create_and_assert_client(self, args):
    argv, remainder = self.parser.parse_known_args(args.split())
    client = self.barbican.create_client(argv)
    self.assertIsNotNone(client)
    return client