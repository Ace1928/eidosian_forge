from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models.execution_environments import ExecutionEnvironment
from awx.main.models.jobs import JobTemplate
from awx.main.tests.functional.conftest import user, system_auditor  # noqa: F401; pylint: disable=unused-import
@pytest.mark.django_db
def test_export_simple(run_module, organization, project, inventory, job_template, scm_credential, machine_credential, workflow_job_template, execution_environment, notification_template, rrule, schedule, admin_user):
    """
    TODO: Ensure there aren't _more_ results in each resource than we expect
    """
    result = run_module('export', dict(all=True), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assets = result['assets']
    u = find_by(assets, 'users', 'username', 'admin')
    assert u['is_superuser'] is True
    find_by(assets, 'organizations', 'name', 'Default')
    r = find_by(assets, 'credentials', 'name', 'scm-cred')
    assert r['credential_type']['kind'] == 'scm'
    assert r['credential_type']['name'] == 'Source Control'
    r = find_by(assets, 'credentials', 'name', 'machine-cred')
    assert r['credential_type']['kind'] == 'ssh'
    assert r['credential_type']['name'] == 'Machine'
    r = find_by(assets, 'job_templates', 'name', 'test-jt')
    assert r['natural_key']['organization']['name'] == 'Default'
    assert r['inventory']['name'] == 'test-inv'
    assert r['project']['name'] == 'test-proj'
    find_by(r['related'], 'credentials', 'name', 'machine-cred')
    r = find_by(assets, 'inventory', 'name', 'test-inv')
    assert r['organization']['name'] == 'Default'
    r = find_by(assets, 'projects', 'name', 'test-proj')
    assert r['organization']['name'] == 'Default'
    r = find_by(assets, 'workflow_job_templates', 'name', 'test-workflow_job_template')
    assert r['natural_key']['organization']['name'] == 'Default'
    assert r['inventory']['name'] == 'test-inv'
    r = find_by(assets, 'execution_environments', 'name', 'test-ee')
    assert r['organization']['name'] == 'Default'
    r = find_by(assets, 'schedules', 'name', 'test-sched')
    assert r['rrule'] == rrule
    r = find_by(assets, 'notification_templates', 'name', 'test-notification_template')
    assert r['organization']['name'] == 'Default'
    assert r['notification_configuration']['url'] == 'http://localhost'