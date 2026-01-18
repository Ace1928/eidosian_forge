from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Label
@pytest.mark.django_db
def test_create_label(run_module, admin_user, organization):
    result = run_module('label', dict(name='test-label', organization=organization.name), admin_user)
    assert not result.get('failed'), result.get('msg', result)
    assert result.get('changed', False)
    assert Label.objects.get(name='test-label').organization == organization