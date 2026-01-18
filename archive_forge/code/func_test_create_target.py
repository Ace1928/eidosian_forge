import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.backup.base import BackupTargetJob
from libcloud.common.types import InvalidCredsError
from libcloud.test.secrets import DIMENSIONDATA_PARAMS
from libcloud.test.file_fixtures import BackupFileFixtures
from libcloud.common.dimensiondata import DimensionDataAPIException
from libcloud.backup.drivers.dimensiondata import DEFAULT_BACKUP_PLAN
from libcloud.backup.drivers.dimensiondata import DimensionDataBackupDriver as DimensionData
def test_create_target(self):
    target = self.driver.create_target('name', 'e75ead52-692f-4314-8725-c8a4f4d13a87', extra={'servicePlan': 'Enterprise'})
    self.assertEqual(target.id, 'ee7c4b64-f7af-4a4f-8384-be362273530f')
    self.assertEqual(target.address, 'e75ead52-692f-4314-8725-c8a4f4d13a87')
    self.assertEqual(target.extra['servicePlan'], 'Enterprise')