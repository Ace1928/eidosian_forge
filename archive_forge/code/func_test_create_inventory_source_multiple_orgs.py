from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Organization, Inventory, InventorySource, Project
@pytest.mark.django_db
def test_create_inventory_source_multiple_orgs(run_module, admin_user):
    org = Organization.objects.create(name='test-org')
    Inventory.objects.create(name='test-inv', organization=org)
    org2 = Organization.objects.create(name='test-org-number-two')
    inv2 = Inventory.objects.create(name='test-inv', organization=org2)
    result = run_module('inventory_source', dict(name='Test Inventory Source', inventory=inv2.name, organization='test-org-number-two', source='ec2', state='present'), admin_user)
    assert result.pop('changed', None), result
    inv_src = InventorySource.objects.get(name='Test Inventory Source')
    assert inv_src.inventory == inv2
    result.pop('invocation')
    assert result == {'name': 'Test Inventory Source', 'id': inv_src.id}