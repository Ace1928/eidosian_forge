from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import CredentialInputSource, Credential, CredentialType
@pytest.fixture
def source_cred_aim_alt(aim_cred_type):
    return Credential.objects.create(name='Alternate CyberArk AIM Cred', credential_type=aim_cred_type, inputs={'url': 'https://cyberark-alt.example.com', 'app_id': 'myAltID', 'verify': 'false'})