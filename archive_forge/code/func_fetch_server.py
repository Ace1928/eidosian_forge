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
def fetch_server(self, server_id):
    """Fetch fresh server object from Nova.

        Log warnings and return None for non-critical API errors.
        Use this method in various ``check_*_complete`` resource methods,
        where intermittent errors can be tolerated.
        """
    server = None
    try:
        server = self.client().servers.get(server_id)
    except exceptions.OverLimit as exc:
        LOG.warning('Received an OverLimit response when fetching server (%(id)s) : %(exception)s', {'id': server_id, 'exception': exc})
    except exceptions.ClientException as exc:
        if getattr(exc, 'http_status', getattr(exc, 'code', None)) in (500, 503):
            LOG.warning('Received the following exception when fetching server (%(id)s) : %(exception)s', {'id': server_id, 'exception': exc})
        else:
            raise
    return server