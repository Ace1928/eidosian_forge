from __future__ import absolute_import, division, print_function
import random
import pytest
from awx.main.models import ActivityStream, JobTemplate, Job, NotificationTemplate, Label
@pytest.mark.django_db
def test_job_template_with_survey_spec(run_module, admin_user, project, inventory, survey_spec):
    result = run_module('job_template', dict(name='foo', playbook='helloworld.yml', project=project.name, inventory=inventory.name, survey_spec=survey_spec, survey_enabled=True), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed', False), result
    jt = JobTemplate.objects.get(pk=result['id'])
    assert jt.survey_spec == survey_spec
    prior_ct = ActivityStream.objects.count()
    result = run_module('job_template', dict(name='foo', playbook='helloworld.yml', project=project.name, inventory=inventory.name, survey_spec=survey_spec, survey_enabled=True), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert not result.get('changed', True), result
    jt.refresh_from_db()
    assert result['id'] == jt.id
    assert jt.survey_spec == survey_spec
    assert ActivityStream.objects.count() == prior_ct