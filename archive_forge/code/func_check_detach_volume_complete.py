import collections
import email
from email.mime import multipart
from email.mime import text
import os
import pkgutil
import string
from urllib import parse as urlparse
from neutronclient.common import exceptions as q_exceptions
from novaclient import api_versions
from novaclient import client as nc
from novaclient import exceptions
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import netutils
import tenacity
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_exception
from heat.engine.clients import client_plugin
from heat.engine.clients import microversion_mixin
from heat.engine.clients import os as os_client
from heat.engine import constraints
def check_detach_volume_complete(self, server_id, attach_id):
    """Check that nova server lost attachment.

        This check is needed for immediate reattachment when updating:
        there might be some time between cinder marking volume as 'available'
        and nova removing attachment from its own objects, so we
        check that nova already knows that the volume is detached.
        """
    try:
        self.client().volumes.get_server_volume(server_id, attach_id)
    except Exception as ex:
        self.ignore_not_found(ex)
        LOG.info('Volume %(vol)s is detached from server %(srv)s', {'vol': attach_id, 'srv': server_id})
        return True
    else:
        LOG.debug('Server %(srv)s still has attachment %(att)s.', {'att': attach_id, 'srv': server_id})
        return False