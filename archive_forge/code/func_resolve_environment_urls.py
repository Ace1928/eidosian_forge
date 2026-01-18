import collections.abc
import json
import typing as ty
from urllib import parse
from urllib import request
from openstack import exceptions
from openstack.orchestration.util import environment_format
from openstack.orchestration.util import template_format
from openstack.orchestration.util import utils
def resolve_environment_urls(resource_registry, files, env_base_url, is_object=False, object_request=None):
    """Handles any resource URLs specified in an environment.

    :param resource_registry: mapping of type name to template filename
    :type  resource_registry: dict
    :param files: dict to store loaded file contents into
    :type  files: dict
    :param env_base_url: base URL to look in when loading files
    :type  env_base_url: str or None
    """
    if resource_registry is None:
        return
    rr = resource_registry
    base_url = rr.get('base_url', env_base_url)

    def ignore_if(key, value):
        if key == 'base_url':
            return True
        if isinstance(value, dict):
            return True
        if '::' in value:
            return True
        if key in ['hooks', 'restricted_actions']:
            return True
    get_file_contents(rr, files, base_url, ignore_if, is_object=is_object, object_request=object_request)
    for res_name, res_dict in rr.get('resources', {}).items():
        res_base_url = res_dict.get('base_url', base_url)
        get_file_contents(res_dict, files, res_base_url, ignore_if, is_object=is_object, object_request=object_request)