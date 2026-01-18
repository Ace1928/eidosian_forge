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
def test_ex_cancel_target_job_with_extras(self):
    success = self.driver.cancel_target_job(None, ex_client='30b1ff76_c76d_4d7c_b39d_3b72be0384c8', ex_target='e75ead52_692f_4314_8725_c8a4f4d13a87')
    self.assertTrue(success)