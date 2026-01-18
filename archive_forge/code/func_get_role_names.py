import uuid
from keystoneauth1 import fixture
from keystoneauth1 import loading
from keystonemiddleware.auth_token import _base
from keystonemiddleware.tests.unit.auth_token import base
def get_role_names(self, token):
    return [x['name'] for x in token['token'].get('roles', [])]