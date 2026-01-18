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
@tenacity.retry(stop=tenacity.stop_after_attempt(cfg.CONF.max_interface_check_attempts), wait=tenacity.wait_fixed(0.5), retry=tenacity.retry_if_result(client_plugin.retry_if_result_is_false))
def check_interface_attach(self, server_id, port_id):
    if not port_id:
        return True
    server = self.fetch_server(server_id)
    if server:
        interfaces = server.interface_list()
        for iface in interfaces:
            if iface.port_id == port_id:
                return True
    return False