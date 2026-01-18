import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_share_instances_1234_export_locations_fake_el_uuid(self, **kw):
    export_location = {'export_location': get_fake_export_location()}
    return (200, {}, export_location)