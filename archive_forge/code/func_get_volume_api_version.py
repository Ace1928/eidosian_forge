from cinderclient import client as cc
from cinderclient import exceptions
from keystoneauth1 import exceptions as ks_exceptions
from oslo_config import cfg
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
from heat.engine import constraints
def get_volume_api_version(self):
    """Returns the most recent API version."""
    self.interface = self._get_client_option(CLIENT_NAME, 'endpoint_type')
    try:
        self.context.keystone_session.get_endpoint(service_type=self.VOLUME_V3, interface=self.interface)
        self.service_type = self.VOLUME_V3
        self.client_version = '3'
    except ks_exceptions.EndpointNotFound:
        raise exception.Error(_('No volume service available.'))