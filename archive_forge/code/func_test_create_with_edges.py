from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import WorkflowJobTemplateNode, WorkflowJobTemplate, JobTemplate, UnifiedJobTemplate
@pytest.mark.django_db
def test_create_with_edges(run_module, admin_user, wfjt, job_template):
    next_nodes = [WorkflowJobTemplateNode.objects.create(identifier='foo{0}'.format(i), workflow_job_template=wfjt, unified_job_template=job_template) for i in range(3)]
    result = run_module('workflow_job_template_node', {'identifier': '42', 'workflow_job_template': 'foo-workflow', 'organization': wfjt.organization.name, 'unified_job_template': 'foo-jt', 'success_nodes': ['foo0'], 'always_nodes': ['foo1'], 'failure_nodes': ['foo2'], 'state': 'present'}, admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed', False)
    node = WorkflowJobTemplateNode.objects.get(identifier='42')
    assert list(node.success_nodes.all()) == [next_nodes[0]]
    assert list(node.always_nodes.all()) == [next_nodes[1]]
    assert list(node.failure_nodes.all()) == [next_nodes[2]]