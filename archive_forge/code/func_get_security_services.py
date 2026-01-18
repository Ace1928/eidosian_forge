import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_security_services(self, **kw):
    security_services = {'security_services': [{'id': 1111, 'name': 'fake_security_service', 'type': 'fake_type', 'status': 'fake_status'}]}
    return (200, {}, security_services)