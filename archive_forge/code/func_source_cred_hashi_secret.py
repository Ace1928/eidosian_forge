from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import CredentialInputSource, Credential, CredentialType
@pytest.fixture
def source_cred_hashi_secret(organization):
    ct = CredentialType.defaults['hashivault_kv']()
    ct.save()
    return Credential.objects.create(name='HashiCorp secret Cred', credential_type=ct, inputs={'url': 'https://secret.hash.example.com', 'token': 'myApiKey', 'role_id': 'role', 'secret_id': 'secret', 'default_auth_path': 'path-to-approle'})