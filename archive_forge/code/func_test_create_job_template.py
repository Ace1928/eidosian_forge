from __future__ import absolute_import, division, print_function
import random
import pytest
from awx.main.models import ActivityStream, JobTemplate, Job, NotificationTemplate, Label
@pytest.mark.django_db
def test_create_job_template(run_module, admin_user, project, inventory):
    module_args = {'name': 'foo', 'playbook': 'helloworld.yml', 'project': project.name, 'inventory': inventory.name, 'extra_vars': {'foo': 'bar'}, 'job_type': 'run', 'state': 'present'}
    result = run_module('job_template', module_args, admin_user)
    jt = JobTemplate.objects.get(name='foo')
    assert jt.extra_vars == '{"foo": "bar"}'
    assert result == {'name': 'foo', 'id': jt.id, 'changed': True, 'invocation': {'module_args': module_args}}
    assert jt.project_id == project.id
    assert jt.inventory_id == inventory.id