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
def create_one_host(attrs=None):
    """Create a fake host.

    :param dict attrs:
        A dictionary with all attributes
    :return:
        A FakeResource object, with uuid and other attributes
    """
    attrs = attrs or {}
    host_info = {'service_id': 1, 'host': 'host1', 'uuid': 'host-id-' + uuid.uuid4().hex, 'vcpus': 10, 'memory_mb': 100, 'local_gb': 100, 'vcpus_used': 5, 'memory_mb_used': 50, 'local_gb_used': 10, 'hypervisor_type': 'xen', 'hypervisor_version': 1, 'hypervisor_hostname': 'devstack1', 'free_ram_mb': 50, 'free_disk_gb': 50, 'current_workload': 10, 'running_vms': 1, 'cpu_info': '', 'disk_available_least': 1, 'host_ip': '10.10.10.10', 'supported_instances': '', 'metrics': '', 'pci_stats': '', 'extra_resources': '', 'stats': '', 'numa_topology': '', 'ram_allocation_ratio': 1.0, 'cpu_allocation_ratio': 1.0, 'zone': 'zone-' + uuid.uuid4().hex, 'host_name': 'name-' + uuid.uuid4().hex, 'service': 'service-' + uuid.uuid4().hex, 'cpu': 4, 'disk_gb': 100, 'project': 'project-' + uuid.uuid4().hex}
    host_info.update(attrs)
    return host_info