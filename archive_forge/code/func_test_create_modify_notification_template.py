from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import NotificationTemplate, Job
@pytest.mark.django_db
def test_create_modify_notification_template(run_module, admin_user, organization):
    nt_config = {'username': 'user', 'password': 'password', 'sender': 'foo@invalid.com', 'recipients': ['foo2@invalid.com'], 'host': 'smtp.example.com', 'port': 25, 'use_tls': False, 'use_ssl': False, 'timeout': 4}
    result = run_module('notification_template', dict(name='foo-notification-template', organization=organization.name, notification_type='email', notification_configuration=nt_config), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.pop('changed', None), result
    nt = NotificationTemplate.objects.get(id=result['id'])
    compare_with_encrypted(nt.notification_configuration, nt_config)
    assert nt.organization == organization
    result = run_module('notification_template', dict(name='foo-notification-template', organization=organization.name, notification_type='email'), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert not result.pop('changed', None), result
    nt_config['timeout'] = 12
    result = run_module('notification_template', dict(name='foo-notification-template', organization=organization.name, notification_type='email', notification_configuration=nt_config), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.pop('changed', None), result
    nt.refresh_from_db()
    compare_with_encrypted(nt.notification_configuration, nt_config)