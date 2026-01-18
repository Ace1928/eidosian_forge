from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Credential, CredentialType, Organization
@pytest.mark.django_db
@pytest.mark.parametrize('state', ('present', 'absent', 'exists'))
def test_credential_state(run_module, organization, admin_user, cred_type, state):
    result = run_module('credential', dict(name='Galaxy Token for Steve', organization=organization.name, credential_type=cred_type.name, inputs={'token': '7rEZK38DJl58A7RxA6EC7lLvUHbBQ1'}, state=state), admin_user)
    assert not result.get('failed', False), result.get('msg', result)