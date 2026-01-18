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
def test_ex_get_backup_details_for_target(self):
    target = self.driver.list_targets()[0]
    response = self.driver.ex_get_backup_details_for_target(target)
    self.assertEqual(response.service_plan, 'Enterprise')
    client = response.clients[0]
    self.assertEqual(client.id, '30b1ff76-c76d-4d7c-b39d-3b72be0384c8')
    self.assertEqual(client.type.type, 'FA.Linux')
    self.assertEqual(client.running_job.progress, 5)
    self.assertTrue(isinstance(client.running_job, BackupTargetJob))
    self.assertEqual(len(client.alert.notify_list), 2)
    self.assertTrue(isinstance(client.alert.notify_list, list))