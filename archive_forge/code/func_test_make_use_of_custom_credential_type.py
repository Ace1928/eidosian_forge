from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Credential, CredentialType, Organization
@pytest.mark.django_db
def test_make_use_of_custom_credential_type(run_module, organization, admin_user, cred_type):
    result = run_module('credential', dict(name='Galaxy Token for Steve', organization=organization.name, credential_type=cred_type.name, inputs={'token': '7rEZK38DJl58A7RxA6EC7lLvUHbBQ1'}), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed', False), result
    cred = Credential.objects.get(name='Galaxy Token for Steve')
    assert cred.credential_type_id == cred_type.id
    assert list(cred.inputs.keys()) == ['token']
    assert cred.inputs['token'].startswith('$encrypted$')
    assert len(cred.inputs['token']) >= len('$encrypted$') + len('7rEZK38DJl58A7RxA6EC7lLvUHbBQ1')
    assert result['name'] == 'Galaxy Token for Steve'
    assert result['id'] == cred.pk