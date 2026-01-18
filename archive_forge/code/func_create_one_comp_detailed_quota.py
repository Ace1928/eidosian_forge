import copy
import random
from unittest import mock
import uuid
from novaclient import api_versions
from openstack.compute.v2 import _proxy
from openstack.compute.v2 import aggregate as _aggregate
from openstack.compute.v2 import availability_zone as _availability_zone
from openstack.compute.v2 import extension as _extension
from openstack.compute.v2 import flavor as _flavor
from openstack.compute.v2 import hypervisor as _hypervisor
from openstack.compute.v2 import keypair as _keypair
from openstack.compute.v2 import migration as _migration
from openstack.compute.v2 import server as _server
from openstack.compute.v2 import server_action as _server_action
from openstack.compute.v2 import server_group as _server_group
from openstack.compute.v2 import server_interface as _server_interface
from openstack.compute.v2 import server_migration as _server_migration
from openstack.compute.v2 import service as _service
from openstack.compute.v2 import usage as _usage
from openstack.compute.v2 import volume_attachment as _volume_attachment
from openstackclient.api import compute_v2
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
def create_one_comp_detailed_quota(attrs=None):
    """Create one quota"""
    attrs = attrs or {}
    quota_attrs = {'id': 'project-id-' + uuid.uuid4().hex, 'cores': {'reserved': 0, 'in_use': 0, 'limit': 20}, 'fixed_ips': {'reserved': 0, 'in_use': 0, 'limit': 30}, 'injected_files': {'reserved': 0, 'in_use': 0, 'limit': 100}, 'injected_file_content_bytes': {'reserved': 0, 'in_use': 0, 'limit': 10240}, 'injected_file_path_bytes': {'reserved': 0, 'in_use': 0, 'limit': 255}, 'instances': {'reserved': 0, 'in_use': 0, 'limit': 50}, 'key_pairs': {'reserved': 0, 'in_use': 0, 'limit': 20}, 'metadata_items': {'reserved': 0, 'in_use': 0, 'limit': 10}, 'ram': {'reserved': 0, 'in_use': 0, 'limit': 51200}, 'server_groups': {'reserved': 0, 'in_use': 0, 'limit': 10}, 'server_group_members': {'reserved': 0, 'in_use': 0, 'limit': 10}}
    quota_attrs.update(attrs)
    quota = fakes.FakeResource(info=copy.deepcopy(quota_attrs), loaded=True)
    quota.project_id = quota_attrs['id']
    return quota