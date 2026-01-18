from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import NotificationTemplate, Job
@pytest.mark.django_db
def test_build_notification_message_undefined(run_module, admin_user, organization):
    """Job notification templates may encounter undefined values in the context when they are
    rendered. Make sure that accessing attributes or items of an undefined value returns another
    instance of Undefined, rather than raising an UndefinedError. This enables the use of expressions
    like "{{ job.created_by.first_name | default('unknown') }}"."""
    job = Job.objects.create(name='foobar')
    nt_config = {'url': 'http://www.example.com/hook', 'headers': {'X-Custom-Header': 'value123'}}
    custom_start_template = {'body': '{"started_by": "{{ job.summary_fields.created_by.username | default(\'My Placeholder\') }}"}'}
    messages = {'started': custom_start_template, 'success': None, 'error': None, 'workflow_approval': None}
    result = run_module('notification_template', dict(name='foo-notification-template', organization=organization.name, notification_type='webhook', notification_configuration=nt_config, messages=messages), admin_user)
    nt = NotificationTemplate.objects.get(id=result['id'])
    body = job.build_notification_message(nt, 'running')
    assert 'The template rendering return a blank body' in body[1]