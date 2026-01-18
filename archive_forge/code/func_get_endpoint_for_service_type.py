import copy
import logging
from openstack.config import loader as config  # noqa
from openstack import connection
from oslo_utils import strutils
from osc_lib.api import auth
from osc_lib import exceptions
def get_endpoint_for_service_type(self, service_type, region_name=None, interface='public'):
    """Return the endpoint URL for the service type."""
    override = self._override_for(service_type)
    if override:
        return override
    if not interface:
        interface = 'public'
    if self.auth_ref:
        endpoint = self.auth_ref.service_catalog.url_for(service_type=service_type, region_name=region_name, interface=interface)
    else:
        endpoint = self.auth.get_endpoint(self.session, interface=interface)
    return endpoint