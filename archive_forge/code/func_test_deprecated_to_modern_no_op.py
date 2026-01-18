from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import NotificationTemplate, Job
@pytest.mark.django_db
def test_deprecated_to_modern_no_op(run_module, admin_user, organization):
    nt_config = {'url': 'http://www.example.com/hook', 'headers': {'X-Custom-Header': 'value123'}}
    result = run_module('notification_template', dict(name='foo-notification-template', organization=organization.name, notification_type='webhook', notification_configuration=nt_config), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.pop('changed', None), result
    result = run_module('notification_template', dict(name='foo-notification-template', organization=organization.name, notification_type='webhook', notification_configuration=nt_config), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert not result.pop('changed', None), result