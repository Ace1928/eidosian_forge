import copy
import datetime
import re
from unittest import mock
from urllib import parse
from oslo_utils import strutils
import novaclient
from novaclient import api_versions
from novaclient import client as base_client
from novaclient import exceptions
from novaclient.tests.unit import fakes
from novaclient.tests.unit import utils
from novaclient.v2 import client
@api_versions.wraps(start_version='2.23')
def get_servers_1234_migrations(self, **kw):
    migrations = {'migrations': [{'created_at': '2016-01-29T13:42:02.000000', 'dest_compute': 'compute2', 'dest_host': '1.2.3.4', 'dest_node': 'node2', 'id': 1, 'server_uuid': '4cfba335-03d8-49b2-8c52-e69043d1e8fe', 'source_compute': 'compute1', 'source_node': 'node1', 'status': 'running', 'memory_total_bytes': 123456, 'memory_processed_bytes': 12345, 'memory_remaining_bytes': 120000, 'disk_total_bytes': 234567, 'disk_processed_bytes': 23456, 'disk_remaining_bytes': 230000, 'updated_at': '2016-01-29T13:42:02.000000'}]}
    if self.api_version >= api_versions.APIVersion('2.80'):
        migrations['migrations'][0].update({'project_id': 'b59c18e5fa284fd384987c5cb25a1853', 'user_id': '13cc0930d27c4be0acc14d7c47a3e1f7'})
    return (200, FAKE_RESPONSE_HEADERS, migrations)