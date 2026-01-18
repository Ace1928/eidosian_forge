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
def get_prometheus_histogram(self):
    registry = self.get_prometheus_registry()
    if not registry or not prometheus_client:
        return
    hist = getattr(registry, '_openstacksdk_histogram', None)
    if not hist:
        hist = prometheus_client.Histogram('openstack_http_response_time', 'Time taken for an http response to an OpenStack service', labelnames=['method', 'endpoint', 'service_type', 'status_code'], registry=registry)
        registry._openstacksdk_histogram = hist
    return hist