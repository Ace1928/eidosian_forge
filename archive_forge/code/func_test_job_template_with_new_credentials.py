from __future__ import absolute_import, division, print_function
import random
import pytest
from awx.main.models import ActivityStream, JobTemplate, Job, NotificationTemplate, Label
@pytest.mark.django_db
def test_job_template_with_new_credentials(run_module, admin_user, project, inventory, machine_credential, vault_credential):
    result = run_module('job_template', dict(name='foo', playbook='helloworld.yml', project=project.name, inventory=inventory.name, credentials=[machine_credential.name, vault_credential.name]), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed', False), result
    jt = JobTemplate.objects.get(pk=result['id'])
    assert set([machine_credential.id, vault_credential.id]) == set([cred.pk for cred in jt.credentials.all()])
    prior_ct = ActivityStream.objects.count()
    result = run_module('job_template', dict(name='foo', playbook='helloworld.yml', project=project.name, inventory=inventory.name, credentials=[machine_credential.name, vault_credential.name]), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert not result.get('changed', True), result
    jt.refresh_from_db()
    assert result['id'] == jt.id
    assert set([machine_credential.id, vault_credential.id]) == set([cred.pk for cred in jt.credentials.all()])
    assert ActivityStream.objects.count() == prior_ct