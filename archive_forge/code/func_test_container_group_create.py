from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import InstanceGroup, Instance
@pytest.mark.django_db
def test_container_group_create(run_module, admin_user, kube_credential):
    pod_spec = "{ 'Nothing': True }"
    result = run_module('instance_group', {'name': 'foo-c-group', 'credential': kube_credential.id, 'is_container_group': True, 'state': 'present'}, admin_user)
    assert not result.get('failed', False), result['msg']
    assert result['changed']
    ig = InstanceGroup.objects.get(name='foo-c-group')
    assert ig.pod_spec_override == ''
    result = run_module('instance_group', {'name': 'foo-c-group', 'credential': kube_credential.id, 'is_container_group': True, 'pod_spec_override': pod_spec, 'state': 'present'}, admin_user)
    assert not result.get('failed', False), result['msg']
    assert result['changed']
    ig = InstanceGroup.objects.get(name='foo-c-group')
    assert ig.pod_spec_override == pod_spec