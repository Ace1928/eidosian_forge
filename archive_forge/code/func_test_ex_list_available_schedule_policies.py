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
def test_ex_list_available_schedule_policies(self):
    target = self.driver.list_targets()[0]
    answer = self.driver.ex_list_available_schedule_policies(target)
    self.assertEqual(len(answer), 1)
    self.assertEqual(answer[0].name, '12AM - 6AM')
    self.assertEqual(answer[0].description, 'Daily backup will start between 12AM - 6AM')