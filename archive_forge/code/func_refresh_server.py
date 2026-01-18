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
def refresh_server(self, server):
    """Refresh server's attributes.

        Also log warnings for non-critical API errors.
        """
    try:
        server.get()
    except exceptions.OverLimit as exc:
        LOG.warning('Server %(name)s (%(id)s) received an OverLimit response during server.get(): %(exception)s', {'name': server.name, 'id': server.id, 'exception': exc})
    except exceptions.ClientException as exc:
        if getattr(exc, 'http_status', getattr(exc, 'code', None)) in (500, 503):
            LOG.warning('Server "%(name)s" (%(id)s) received the following exception during server.get(): %(exception)s', {'name': server.name, 'id': server.id, 'exception': exc})
        else:
            raise