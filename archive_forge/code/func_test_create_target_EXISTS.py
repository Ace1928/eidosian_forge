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
def test_create_target_EXISTS(self):
    DimensionDataMockHttp.type = 'EXISTS'
    with self.assertRaises(DimensionDataAPIException) as context:
        self.driver.create_target('name', 'e75ead52-692f-4314-8725-c8a4f4d13a87', extra={'servicePlan': 'Enterprise'})
    self.assertEqual(context.exception.code, 'ERROR')
    self.assertEqual(context.exception.msg, 'Cloud backup for this server is already enabled or being enabled (state: NORMAL).')