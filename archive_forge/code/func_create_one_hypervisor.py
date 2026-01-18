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
def create_one_hypervisor(attrs=None):
    """Create a fake hypervisor.

    :param dict attrs: A dictionary with all attributes
    :return: A fake openstack.compute.v2.hypervisor.Hypervisor object
    """
    attrs = attrs or {}
    hypervisor_info = {'id': 'hypervisor-id-' + uuid.uuid4().hex, 'hypervisor_hostname': 'hypervisor-hostname-' + uuid.uuid4().hex, 'status': 'enabled', 'host_ip': '192.168.0.10', 'cpu_info': {'aaa': 'aaa'}, 'free_disk_gb': 50, 'hypervisor_version': 2004001, 'disk_available_least': 50, 'local_gb': 50, 'free_ram_mb': 1024, 'service': {'host': 'aaa', 'disabled_reason': None, 'id': 1}, 'vcpus_used': 0, 'hypervisor_type': 'QEMU', 'local_gb_used': 0, 'vcpus': 4, 'memory_mb_used': 512, 'memory_mb': 1024, 'current_workload': 0, 'state': 'up', 'running_vms': 0}
    hypervisor_info.update(attrs)
    hypervisor = _hypervisor.Hypervisor(**hypervisor_info, loaded=True)
    return hypervisor