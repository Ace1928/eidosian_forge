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
def test_list_targets(self):
    targets = self.driver.list_targets()
    self.assertEqual(len(targets), 2)
    self.assertEqual(targets[0].id, '5579f3a7-4c32-4cf5-8a7e-b45c36a35c10')
    self.assertEqual(targets[0].address, 'e75ead52-692f-4314-8725-c8a4f4d13a87')
    self.assertEqual(targets[0].extra['servicePlan'], 'Enterprise')