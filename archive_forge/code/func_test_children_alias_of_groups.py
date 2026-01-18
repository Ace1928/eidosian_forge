from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Organization, Inventory, Group, Host
@pytest.mark.django_db
def test_children_alias_of_groups(run_module, admin_user, organization):
    inv = Inventory.objects.create(name='test-inv', organization=organization)
    group = Group.objects.create(name='Test Group', inventory=inv)
    child = Group.objects.create(inventory=inv, name='child_group')
    result = run_module('group', dict(name='Test Group', inventory='test-inv', groups=[child.name], state='present'), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result['changed'] is True
    assert set(group.children.all()) == set([child])