import functools
import typing as ty
import urllib
from urllib.parse import urlparse
import iso8601
import jmespath
from keystoneauth1 import adapter
from openstack import _log
from openstack import exceptions
from openstack import resource
def should_skip_resource_cleanup(self, resource=None, skip_resources=None):
    if resource is None or skip_resources is None:
        return False
    resource_name = f'{self.service_type.replace('-', '_')}.{resource}'
    if resource_name in skip_resources:
        self.log.debug(f'Skipping resource {resource_name} in project cleanup')
        return True
    return False