import copy
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
import testtools
import barbicanclient.barbican
def generate_v3_project_scoped_token(**kwargs):
    """Generate a Keystone V3 token based on auth request."""
    ref = _get_normalized_token_data(**kwargs)
    o = {'token': {'expires_at': '2099-05-22T00:02:43.941430Z', 'issued_at': '2013-05-21T00:02:43.941473Z', 'methods': ['password'], 'project': {'id': ref.get('project_id'), 'name': ref.get('project_name'), 'domain': {'id': ref.get('project_domain_id'), 'name': ref.get('project_domain_name')}}, 'user': {'id': ref.get('user_id'), 'name': ref.get('username'), 'domain': {'id': ref.get('user_domain_id'), 'name': ref.get('user_domain_name')}}, 'roles': ref.get('roles')}}
    o['token']['catalog'] = [{'endpoints': [{'id': uuidutils.generate_uuid(dashed=False), 'interface': 'public', 'region': 'RegionTwo', 'url': ref.get('barbican_url')}], 'id': uuidutils.generate_uuid(dashed=False), 'type': 'keystore'}, {'endpoints': [{'id': uuidutils.generate_uuid(dashed=False), 'interface': 'public', 'region': 'RegionTwo', 'url': ref.get('auth_url')}, {'id': uuidutils.generate_uuid(dashed=False), 'interface': 'admin', 'region': 'RegionTwo', 'url': ref.get('auth_url')}], 'id': uuidutils.generate_uuid(dashed=False), 'type': 'identity'}]
    token_id = uuidutils.generate_uuid(dashed=False)
    return (token_id, o)