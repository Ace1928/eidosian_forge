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
def get_os_migrations(self, **kw):
    migration1 = {'created_at': '2012-10-29T13:42:02.000000', 'dest_compute': 'compute2', 'dest_host': '1.2.3.4', 'dest_node': 'node2', 'id': '1234', 'instance_uuid': 'instance_id_123', 'new_instance_type_id': 2, 'old_instance_type_id': 1, 'source_compute': 'compute1', 'source_node': 'node1', 'status': 'Done', 'updated_at': '2012-10-29T13:42:02.000000'}
    migration2 = {'created_at': '2012-10-29T13:42:02.000000', 'dest_compute': 'compute2', 'dest_host': '1.2.3.4', 'dest_node': 'node2', 'id': '1234', 'instance_uuid': 'instance_id_456', 'new_instance_type_id': 2, 'old_instance_type_id': 1, 'source_compute': 'compute1', 'source_node': 'node1', 'status': 'Done', 'updated_at': '2013-11-50T13:42:02.000000'}
    if self.api_version >= api_versions.APIVersion('2.23'):
        migration1.update({'migration_type': 'live-migration'})
        migration2.update({'migration_type': 'live-migration'})
    if self.api_version >= api_versions.APIVersion('2.59'):
        migration1.update({'uuid': '11111111-07d5-11e1-90e3-e3dffe0c5983'})
        migration2.update({'uuid': '22222222-07d5-11e1-90e3-e3dffe0c5983'})
    if self.api_version >= api_versions.APIVersion('2.80'):
        migration1.update({'project_id': 'b59c18e5fa284fd384987c5cb25a1853', 'user_id': '13cc0930d27c4be0acc14d7c47a3e1f7'})
        migration2.update({'project_id': 'b59c18e5fa284fd384987c5cb25a1853', 'user_id': '13cc0930d27c4be0acc14d7c47a3e1f7'})
    migration_list = []
    instance_uuid = kw.get('instance_uuid', None)
    if instance_uuid == migration1['instance_uuid']:
        migration_list.append(migration1)
    elif instance_uuid == migration2['instance_uuid']:
        migration_list.append(migration2)
    elif instance_uuid is None:
        migration_list.extend([migration1, migration2])
    migrations = {'migrations': migration_list}
    return (200, FAKE_RESPONSE_HEADERS, migrations)