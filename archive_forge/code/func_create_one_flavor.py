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
def create_one_flavor(attrs=None):
    """Create a fake flavor.

    :param dict attrs: A dictionary with all attributes
    :return: A fake openstack.compute.v2.flavor.Flavor object
    """
    attrs = attrs or {}
    flavor_info = {'id': 'flavor-id-' + uuid.uuid4().hex, 'name': 'flavor-name-' + uuid.uuid4().hex, 'ram': 8192, 'vcpus': 4, 'disk': 128, 'swap': 0, 'rxtx_factor': 1.0, 'OS-FLV-DISABLED:disabled': False, 'os-flavor-access:is_public': True, 'description': 'description', 'OS-FLV-EXT-DATA:ephemeral': 0, 'extra_specs': {'property': 'value'}}
    flavor_info.update(attrs)
    flavor = _flavor.Flavor(**flavor_info)
    return flavor