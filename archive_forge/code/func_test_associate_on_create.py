from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Organization, Inventory, Group, Host
@pytest.mark.django_db
def test_associate_on_create(run_module, admin_user, organization):
    inv = Inventory.objects.create(name='test-inv', organization=organization)
    child = Group.objects.create(name='test-child', inventory=inv)
    host = Host.objects.create(name='test-host', inventory=inv)
    result = run_module('group', dict(name='Test Group', inventory='test-inv', hosts=[host.name], groups=[child.name], state='present'), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result['changed'] is True
    group = Group.objects.get(pk=result['id'])
    assert set(group.hosts.all()) == set([host])
    assert set(group.children.all()) == set([child])