from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Project
@pytest.mark.django_db
def test_create_manual_project(run_module, admin_user, organization, mocker):
    mocker.patch('awx.main.models.projects.Project.get_local_path_choices', return_value=['foo_folder/'])
    result = run_module('project', dict(name='foo', organization=organization.name, scm_type='manual', local_path='foo_folder/', wait=False), admin_user)
    assert result.pop('changed', None), result
    proj = Project.objects.get(name='foo')
    assert proj.local_path == 'foo_folder/'
    assert proj.organization == organization
    result.pop('invocation')
    assert result == {'name': 'foo', 'id': proj.id}