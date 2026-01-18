from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Organization, Inventory, InventorySource, Project
@pytest.mark.django_db
def test_falsy_value(run_module, admin_user, base_inventory):
    result = run_module('inventory_source', dict(name='falsy-test', inventory=base_inventory.name, source='ec2', update_on_launch=True), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed', None), result
    inv_src = InventorySource.objects.get(name='falsy-test')
    assert inv_src.update_on_launch is True
    result = run_module('inventory_source', dict(name='falsy-test', inventory=base_inventory.name, source='ec2', update_on_launch=False), admin_user)
    inv_src.refresh_from_db()
    assert inv_src.update_on_launch is False