from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Credential, CredentialType, Organization
@pytest.fixture
def cred_type():
    ct = CredentialType.objects.create(name='Ansible Galaxy Token', inputs={'fields': [{'id': 'token', 'type': 'string', 'secret': True, 'label': 'Ansible Galaxy Secret Token Value'}], 'required': ['token']}, injectors={'extra_vars': {'galaxy_token': '{{token}}'}})
    return ct