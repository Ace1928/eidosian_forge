from random import randint
import ddt
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
@utils.skip_if_microversion_not_supported('2.38')
def test_delete_share_type_quotas_with_too_old_microversion(self):
    cmd = 'quota-delete --tenant-id %s --share-type %s' % (self.project_id, self.st_id)
    self.assertRaises(exceptions.CommandFailed, self.admin_client.manila, cmd, microversion='2.38')