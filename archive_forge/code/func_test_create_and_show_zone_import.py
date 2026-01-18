from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_zone_file
from designateclient.functionaltests.v2.fixtures import ImportFixture
def test_create_and_show_zone_import(self):
    zone_import = self.useFixture(ImportFixture(zone_file_contents=self.zone_file_contents)).zone_import
    fetched_import = self.clients.zone_import_show(zone_import.id)
    self.assertEqual(zone_import.created_at, fetched_import.created_at)
    self.assertEqual(zone_import.id, fetched_import.id)
    self.assertEqual(zone_import.project_id, fetched_import.project_id)
    self.assertIn(fetched_import.status, ['PENDING', 'COMPLETE'])