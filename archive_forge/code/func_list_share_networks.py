import ast
import re
import time
from oslo_utils import strutils
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import exceptions
from manilaclient.tests.functional import utils
def list_share_networks(self, all_tenants=False, filters=None, columns=None, microversion=None):
    """List share networks.

        :param all_tenants: bool -- whether to list share-networks that belong
            only to current project or for all tenants.
        :param filters: dict -- filters for listing of share networks.
            Example, input:
                {'project_id': 'foo'}
                {'-project_id': 'foo'}
                {'--project_id': 'foo'}
                {'project-id': 'foo'}
            will be transformed to filter parameter "--project-id=foo"
         :param columns: comma separated string of columns.
            Example, "--columns id"
        """
    cmd = 'share-network-list '
    if columns is not None:
        cmd += ' --columns ' + columns
    if all_tenants:
        cmd += ' --all-tenants '
    if filters and isinstance(filters, dict):
        for k, v in filters.items():
            cmd += '%(k)s=%(v)s ' % {'k': self._stranslate_to_cli_optional_param(k), 'v': v}
    share_networks_raw = self.manila(cmd, microversion=microversion)
    share_networks = utils.listing(share_networks_raw)
    return share_networks