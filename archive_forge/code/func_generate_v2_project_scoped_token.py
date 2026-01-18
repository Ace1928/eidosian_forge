import copy
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
import testtools
import barbicanclient.barbican
def generate_v2_project_scoped_token(**kwargs):
    """Generate a Keystone V2 token based on auth request."""
    ref = _get_normalized_token_data(**kwargs)
    o = {'access': {'token': {'id': uuidutils.generate_uuid(dashed=False), 'expires': '2099-05-22T00:02:43.941430Z', 'issued_at': '2013-05-21T00:02:43.941473Z', 'tenant': {'enabled': True, 'id': ref.get('project_id'), 'name': ref.get('project_id')}}, 'user': {'id': ref.get('user_id'), 'name': uuidutils.generate_uuid(dashed=False), 'username': ref.get('username'), 'roles': ref.get('roles'), 'roles_links': ref.get('roles_links')}}}
    o['access']['serviceCatalog'] = [{'endpoints': [{'publicURL': ref.get('barbican_url'), 'id': uuidutils.generate_uuid(dashed=False), 'region': 'RegionOne'}], 'endpoints_links': [], 'name': 'Barbican', 'type': 'keystore'}, {'endpoints': [{'publicURL': ref.get('auth_url'), 'adminURL': ref.get('auth_url'), 'id': uuidutils.generate_uuid(dashed=False), 'region': 'RegionOne'}], 'endpoint_links': [], 'name': 'keystone', 'type': 'identity'}]
    return o