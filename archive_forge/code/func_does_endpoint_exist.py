import abc
import weakref
from keystoneauth1 import exceptions
from keystoneauth1.identity import generic
from keystoneauth1 import plugin
from oslo_config import cfg
from oslo_utils import excutils
import requests
from heat.common import config
from heat.common import exception as heat_exception
def does_endpoint_exist(self, service_type, service_name):
    endpoint_key = (service_type, service_name)
    if endpoint_key not in self._endpoint_existence:
        endpoint_type = self._get_client_option(service_name, 'endpoint_type')
        try:
            self.url_for(service_type=service_type, endpoint_type=endpoint_type)
            self._endpoint_existence[endpoint_key] = True
        except exceptions.EndpointNotFound:
            self._endpoint_existence[endpoint_key] = False
    return self._endpoint_existence[endpoint_key]