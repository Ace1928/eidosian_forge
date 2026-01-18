from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_instance_export_locations
def test_get_single_export_location(self):
    share_instance_id = '1234'
    el_uuid = 'fake_el_uuid'
    cs.share_instance_export_locations.get(share_instance_id, el_uuid)
    cs.assert_called('GET', '/share_instances/%(share_instance_id)s/export_locations/%(el_uuid)s' % {'share_instance_id': share_instance_id, 'el_uuid': el_uuid})