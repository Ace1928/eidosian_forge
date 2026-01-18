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
def check_resize(self, server_id, flavor):
    """Verify that a resizing server is properly resized.

        If that's the case, confirm the resize, if not raise an error.
        """
    server = self.fetch_server(server_id)
    if not server or server.status in ('RESIZE', 'ACTIVE'):
        return False
    if server.status == 'VERIFY_RESIZE':
        return True
    else:
        raise exception.Error(_("Resizing to '%(flavor)s' failed, status '%(status)s'") % dict(flavor=flavor, status=server.status))