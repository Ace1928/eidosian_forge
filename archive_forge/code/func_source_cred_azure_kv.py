from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import CredentialInputSource, Credential, CredentialType
@pytest.fixture
def source_cred_azure_kv(organization):
    ct = CredentialType.defaults['azure_kv']()
    ct.save()
    return Credential.objects.create(name='Azure KV Cred', credential_type=ct, inputs={'url': 'https://key.azure.example.com', 'client': 'client', 'secret': 'secret', 'tenant': 'tenant', 'cloud_name': 'the_cloud'})