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
def list_share_instances(self, share_id=None, filters=None, microversion=None):
    """List share instances.

        :param share_id: ID of a share to filter by.
        :param filters: dict -- filters for listing of shares.
            Example, input:
                {'project_id': 'foo'}
                {-'project_id': 'foo'}
                {--'project_id': 'foo'}
                {'project-id': 'foo'}
            will be transformed to filter parameter "--export-location=foo"
        :param microversion: API microversion to be used for request.
        """
    cmd = 'share-instance-list '
    if share_id:
        cmd += '--share-id %s' % share_id
    if filters and isinstance(filters, dict):
        for k, v in filters.items():
            cmd += '%(k)s=%(v)s ' % {'k': self._stranslate_to_cli_optional_param(k), 'v': v}
    share_instances_raw = self.manila(cmd, microversion=microversion)
    share_instances = utils.listing(share_instances_raw)
    return share_instances