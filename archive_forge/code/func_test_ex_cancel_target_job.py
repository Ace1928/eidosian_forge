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
def test_ex_cancel_target_job(self):
    target = self.driver.list_targets()[0]
    response = self.driver.ex_get_backup_details_for_target(target)
    client = response.clients[0]
    self.assertTrue(isinstance(client.running_job, BackupTargetJob))
    success = client.running_job.cancel()
    self.assertTrue(success)