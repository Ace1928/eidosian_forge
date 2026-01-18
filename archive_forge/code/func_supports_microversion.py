from unittest import mock
from keystoneauth1 import discover
from openstack.block_storage.v3 import group as _group
from openstack.block_storage.v3 import group_snapshot as _group_snapshot
from openstack.test import fakes as sdk_fakes
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group_snapshot
def supports_microversion(adapter, microversion, raise_exception=False):
    required = discover.normalize_version_number(microversion)
    candidate = discover.normalize_version_number(mocked_version)
    return discover.version_match(required, candidate)