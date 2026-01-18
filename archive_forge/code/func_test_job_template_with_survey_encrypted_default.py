from __future__ import absolute_import, division, print_function
import random
import pytest
from awx.main.models import ActivityStream, JobTemplate, Job, NotificationTemplate, Label
@pytest.mark.django_db
def test_job_template_with_survey_encrypted_default(run_module, admin_user, project, inventory, silence_warning):
    spec = {'spec': [{'index': 0, 'question_name': 'my question?', 'default': 'very_secret_value', 'variable': 'myvar', 'type': 'password', 'required': False}], 'description': 'test', 'name': 'test'}
    for i in range(2):
        result = run_module('job_template', dict(name='foo', playbook='helloworld.yml', project=project.name, inventory=inventory.name, survey_spec=spec, survey_enabled=True), admin_user)
        assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed', False), result
    silence_warning.assert_called_once_with('The field survey_spec of job_template {0} has encrypted data and may inaccurately report task is changed.'.format(result['id']))