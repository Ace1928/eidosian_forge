from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import CredentialInputSource, Credential, CredentialType
@pytest.fixture
def aim_cred_type():
    ct = CredentialType.defaults['aim']()
    ct.save()
    return ct