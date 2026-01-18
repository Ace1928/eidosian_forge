from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Organization, Inventory, InventorySource, Project
@pytest.mark.django_db
def test_inventory_source_create(run_module, admin_user, base_inventory, project):
    source_path = '/var/lib/awx/example_source_path/'
    result = run_module('inventory_source', dict(name='foo', inventory=base_inventory.name, state='present', source='scm', source_path=source_path, source_project=project.name), admin_user)
    assert result.pop('changed', None), result
    inv_src = InventorySource.objects.get(name='foo')
    assert inv_src.inventory == base_inventory
    result.pop('invocation')
    assert result == {'id': inv_src.id, 'name': 'foo'}