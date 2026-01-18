from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_os_availability_zone_detail(self, **kw):
    return (200, {}, {'availabilityZoneInfo': [{'zoneName': 'zone-1', 'zoneState': {'available': True}, 'hosts': {'fake_host-1': {'cinder-volume': {'active': True, 'available': True, 'updated_at': datetime(2012, 12, 26, 14, 45, 25, 0)}}}}, {'zoneName': 'internal', 'zoneState': {'available': True}, 'hosts': {'fake_host-1': {'cinder-sched': {'active': True, 'available': True, 'updated_at': datetime(2012, 12, 26, 14, 45, 24, 0)}}}}, {'zoneName': 'zone-2', 'zoneState': {'available': False}, 'hosts': None}]})