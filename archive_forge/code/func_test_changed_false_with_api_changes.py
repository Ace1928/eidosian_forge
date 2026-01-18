from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import CredentialType
@pytest.mark.django_db
def test_changed_false_with_api_changes(run_module, admin_user):
    result = run_module('credential_type', dict(name='foo', kind='cloud', inputs={'fields': [{'id': 'env_value', 'label': 'foo', 'default': 'foo'}]}, injectors={'env': {'TEST_ENV_VAR': '{{ env_value }}'}}), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed'), result
    result = run_module('credential_type', dict(name='foo', inputs={'fields': [{'id': 'env_value', 'label': 'foo', 'default': 'foo'}]}, injectors={'env': {'TEST_ENV_VAR': '{{ env_value }}'}}), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert not result.get('changed'), result