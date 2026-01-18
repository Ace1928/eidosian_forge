from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_os_services(self, **kw):
    host = kw.get('host', None)
    binary = kw.get('binary', None)
    services = [{'binary': 'cinder-volume', 'host': 'host1', 'zone': 'cinder', 'status': 'enabled', 'state': 'up', 'updated_at': datetime(2012, 10, 29, 13, 42, 2)}, {'binary': 'cinder-volume', 'host': 'host2', 'zone': 'cinder', 'status': 'disabled', 'state': 'down', 'updated_at': datetime(2012, 9, 18, 8, 3, 38)}, {'binary': 'cinder-scheduler', 'host': 'host2', 'zone': 'cinder', 'status': 'disabled', 'state': 'down', 'updated_at': datetime(2012, 9, 18, 8, 3, 38)}]
    if host:
        services = [i for i in services if i['host'] == host]
    if binary:
        services = [i for i in services if i['binary'] == binary]
    return (200, {}, {'services': services})