from unittest import mock
from glance_store._drivers.swift import connection_manager
from glance_store._drivers.swift import store as swift_store
from glance_store import exceptions
from glance_store.tests import base
def test_multi_tenant_client_cm_with_client_creation_fails(self):
    store = self.prepare_store(multi_tenant=True)
    store.init_client.side_effect = [Exception]
    manager = connection_manager.MultiTenantConnectionManager(store=store, store_location=self.location, context=self.context, allow_reauth=True)
    store.init_client.assert_called_once_with(self.location, self.context)
    store.get_store_connection.assert_called_once_with(self.context.auth_token, manager.storage_url)
    self.assertFalse(manager.allow_reauth)