from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Credential, CredentialType, Organization
@pytest.mark.django_db
def test_create_vault_credential(run_module, admin_user, organization):
    Organization.objects.create(name='test-org')
    ct = CredentialType.defaults['vault']()
    ct.save()
    result = run_module('credential', dict(name='Test Vault Credential', organization=organization.name, credential_type='Vault', inputs={'vault_id': 'bar', 'vault_password': 'foobar'}, state='present'), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed'), result
    cred = Credential.objects.get(name='Test Vault Credential')
    assert cred.credential_type == ct
    assert 'vault_id' in cred.inputs
    assert 'vault_password' in cred.inputs
    assert result['name'] == 'Test Vault Credential'
    assert result['id'] == cred.pk