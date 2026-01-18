from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import ExportFixture
from designateclient.functionaltests.v2.fixtures import ZoneFixture
def test_list_zone_exports(self):
    zone_export = self.useFixture(ExportFixture(zone=self.zone)).zone_export
    zone_exports = self.clients.zone_export_list()
    self.assertGreater(len(zone_exports), 0)
    self.assertTrue(self._is_entity_in_list(zone_export, zone_exports))