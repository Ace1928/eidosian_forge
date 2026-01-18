import os
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import uuidutils
from glance.common import exception
from glance.common import store_utils
from glance import context
from glance.i18n import _LE
def staging_store_path():
    """Return the local path to the staging store.

    :raises: GlanceException if staging store is not configured to be
             a file:// URI
    """
    if CONF.enabled_backends:
        separator, staging_dir = store_utils.get_dir_separator()
    else:
        staging_dir = CONF.node_staging_uri
    expected_prefix = 'file://'
    if not staging_dir.startswith(expected_prefix):
        raise exception.GlanceException('Unexpected scheme in staging store; unable to scan for residue')
    return staging_dir[len(expected_prefix):]