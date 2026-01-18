from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Organization, Inventory, InventorySource, Project
@pytest.mark.django_db
def test_create_inventory_source_implied_org(run_module, admin_user):
    org = Organization.objects.create(name='test-org')
    inv = Inventory.objects.create(name='test-inv', organization=org)
    result = run_module('inventory_source', dict(name='Test Inventory Source', inventory='test-inv', source='ec2', state='present'), admin_user)
    assert result.pop('changed', None), result
    inv_src = InventorySource.objects.get(name='Test Inventory Source')
    assert inv_src.inventory == inv
    result.pop('invocation')
    assert result == {'name': 'Test Inventory Source', 'id': inv_src.id}