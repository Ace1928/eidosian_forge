from unittest import mock
from glance_store._drivers.swift import connection_manager
from glance_store._drivers.swift import store as swift_store
from glance_store import exceptions
from glance_store.tests import base
def test_basic_single_tenant_cm_init(self):
    store = self.prepare_store()
    manager = connection_manager.SingleTenantConnectionManager(store=store, store_location=self.location)
    store.init_client.assert_called_once_with(self.location, None)
    self.client.session.get_endpoint.assert_called_once_with(service_type=store.service_type, interface=store.endpoint_type, region_name=store.region)
    store.get_store_connection.assert_called_once_with('fake_token', manager.storage_url)