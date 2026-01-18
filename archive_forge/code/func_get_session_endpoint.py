import copy
import os.path
import typing as ty
from urllib import parse
import warnings
from keystoneauth1 import discover
import keystoneauth1.exceptions.catalog
from keystoneauth1.loading import adapter as ks_load_adap
from keystoneauth1 import session as ks_session
import os_service_types
import requestsexceptions
from openstack import _log
from openstack.config import _util
from openstack.config import defaults as config_defaults
from openstack import exceptions
from openstack import proxy
from openstack import version as openstack_version
from openstack import warnings as os_warnings
def get_session_endpoint(self, service_type, min_version=None, max_version=None):
    """Return the endpoint from config or the catalog.

        If a configuration lists an explicit endpoint for a service,
        return that. Otherwise, fetch the service catalog from the
        keystone session and return the appropriate endpoint.

        :param service_type: Official service type of service
        """
    override_endpoint = self.get_endpoint(service_type)
    if override_endpoint:
        return override_endpoint
    region_name = self.get_region_name(service_type)
    service_name = self.get_service_name(service_type)
    interface = self.get_interface(service_type)
    session = self.get_session()
    version_kwargs = {}
    if min_version:
        version_kwargs['min_version'] = min_version
    if max_version:
        version_kwargs['max_version'] = max_version
    try:
        endpoint = session.get_endpoint(service_type=service_type, region_name=region_name, interface=interface, service_name=service_name, **version_kwargs)
    except keystoneauth1.exceptions.catalog.EndpointNotFound:
        endpoint = None
    if not endpoint:
        self.log.warning('Keystone catalog entry not found (service_type=%s,service_name=%s,interface=%s,region_name=%s)', service_type, service_name, interface, region_name)
    return endpoint