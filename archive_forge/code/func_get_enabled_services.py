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
def get_enabled_services(self):
    services = set()
    all_services = [k['service_type'] for k in self._service_type_manager.services]
    all_services.extend((k[4:] for k in self.config.keys() if k.startswith('has_')))
    for srv in all_services:
        ep = self.get_endpoint_from_catalog(srv)
        if ep:
            services.add(srv.replace('-', '_'))
    return services