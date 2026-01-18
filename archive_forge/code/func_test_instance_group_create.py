from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import InstanceGroup, Instance
@pytest.mark.django_db
def test_instance_group_create(run_module, admin_user):
    result = run_module('instance_group', {'name': 'foo-group', 'policy_instance_percentage': 34, 'policy_instance_minimum': 12, 'state': 'present'}, admin_user)
    assert not result.get('failed', False), result
    assert result['changed']
    ig = InstanceGroup.objects.get(name='foo-group')
    assert ig.policy_instance_percentage == 34
    assert ig.policy_instance_minimum == 12
    new_instance = Instance.objects.create(hostname='foo.example.com')
    result = run_module('instance_group', {'name': 'foo-group', 'instances': [new_instance.hostname], 'state': 'present'}, admin_user)
    assert not result.get('failed', False), result
    assert result['changed']
    ig = InstanceGroup.objects.get(name='foo-group')
    all_instance_names = []
    for instance in ig.instances.all():
        all_instance_names.append(instance.hostname)
    assert new_instance.hostname in all_instance_names, 'Failed to add instance to group'
    assert len(all_instance_names) == 1, 'Too many instances in group {0}'.format(','.join(all_instance_names))