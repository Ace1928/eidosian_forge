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
def share_server_migration_start(self, server_id, dest_host, writable=False, nondisruptive=False, preserve_snapshots=False, new_share_network=None):
    cmd = 'share-server-migration-start %(server_id)s %(host)s --writable %(writable)s --nondisruptive %(nondisruptive)s --preserve-snapshots %(preserve_snapshots)s' % {'server_id': server_id, 'host': dest_host, 'writable': writable, 'nondisruptive': nondisruptive, 'preserve_snapshots': preserve_snapshots}
    if new_share_network:
        cmd += ' --new-share-network %s' % new_share_network
    return self.manila(cmd)