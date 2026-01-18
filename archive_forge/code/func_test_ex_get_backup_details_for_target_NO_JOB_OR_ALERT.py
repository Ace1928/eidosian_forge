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
def test_ex_get_backup_details_for_target_NO_JOB_OR_ALERT(self):
    DimensionDataMockHttp.type = 'NOJOB'
    response = self.driver.ex_get_backup_details_for_target('e75ead52-692f-4314_8725-c8a4f4d13a87')
    self.assertEqual(response.service_plan, 'Enterprise')
    self.assertTrue(isinstance(response.clients, list))
    self.assertEqual(len(response.clients), 1)
    client = response.clients[0]
    self.assertEqual(client.id, '30b1ff76-c76d-4d7c-b39d-3b72be0384c8')
    self.assertEqual(client.type.type, 'FA.Linux')
    self.assertIsNone(client.running_job)
    self.assertIsNone(client.alert)