from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_zone_file
from designateclient.functionaltests.v2.fixtures import ImportFixture
def test_delete_zone_import(self):
    zone_import = self.useFixture(ImportFixture(zone_file_contents=self.zone_file_contents)).zone_import
    zone_imports = self.clients.zone_import_list()
    self.assertTrue(self._is_entity_in_list(zone_import, zone_imports))
    self.clients.zone_import_delete(zone_import.id)
    zone_imports = self.clients.zone_import_list()
    self.assertFalse(self._is_entity_in_list(zone_import, zone_imports))