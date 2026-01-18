from glance_store import capabilities as caps
from glance_store.tests import base
def test_mixed_capabilities_setup(self):
    self._verify_store_capabilities(FakeStoreWithMixedCapabilities())