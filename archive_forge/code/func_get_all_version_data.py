import abc
import base64
import functools
import hashlib
import json
import threading
from keystoneauth1 import _utils as utils
from keystoneauth1 import access
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
def get_all_version_data(self, session, interface='public', region_name=None, service_type=None, **kwargs):
    """Get version data for all services in the catalog.

        :param session: A session object that can be used for communication.
        :type session: keystoneauth1.session.Session
        :param interface:
            Type of endpoint to get version data for. Can be a single value
            or a list of values. A value of None indicates that all interfaces
            should be queried. (optional, defaults to public)
        :param string region_name:
            Region of endpoints to get version data for. A valueof None
            indicates that all regions should be queried. (optional, defaults
            to None)
        :param string service_type:
            Limit the version data to a single service. (optional, defaults
            to None)
        :returns: A dictionary keyed by region_name with values containing
            dictionaries keyed by interface with values being a list of
            :class:`~keystoneauth1.discover.VersionData`.
        """
    service_types = discover._SERVICE_TYPES
    catalog = self.get_access(session).service_catalog
    version_data = {}
    endpoints_data = catalog.get_endpoints_data(interface=interface, region_name=region_name, service_type=service_type)
    for endpoint_service_type, services in endpoints_data.items():
        if service_types.is_known(endpoint_service_type):
            endpoint_service_type = service_types.get_service_type(endpoint_service_type)
        for service in services:
            versions = service.get_all_version_string_data(session=session, project_id=self.get_project_id(session))
            if service.region_name not in version_data:
                version_data[service.region_name] = {}
            regions = version_data[service.region_name]
            interface = service.interface.rstrip('URL')
            if interface not in regions:
                regions[interface] = {}
            regions[interface][endpoint_service_type] = versions
    return version_data