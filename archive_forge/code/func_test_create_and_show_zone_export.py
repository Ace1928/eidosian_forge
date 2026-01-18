from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import ExportFixture
from designateclient.functionaltests.v2.fixtures import ZoneFixture
def test_create_and_show_zone_export(self):
    zone_export = self.useFixture(ExportFixture(zone=self.zone)).zone_export
    fetched_export = self.clients.zone_export_show(zone_export.id)
    self.assertEqual(zone_export.created_at, fetched_export.created_at)
    self.assertEqual(zone_export.id, fetched_export.id)
    self.assertEqual(zone_export.message, fetched_export.message)
    self.assertEqual(zone_export.project_id, fetched_export.project_id)
    self.assertEqual(zone_export.zone_id, fetched_export.zone_id)