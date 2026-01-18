import collections.abc
import json
import typing as ty
from urllib import parse
from urllib import request
from openstack import exceptions
from openstack.orchestration.util import environment_format
from openstack.orchestration.util import template_format
from openstack.orchestration.util import utils
def process_environment_and_files(env_path=None, template=None, template_url=None, env_path_is_object=None, object_request=None, include_env_in_files=False):
    """Loads a single environment file.

    Returns an entry suitable for the files dict which maps the environment
    filename to its contents.

    :param env_path: full path to the file to load
    :type  env_path: str or None
    :param include_env_in_files: if specified, the raw environment file itself
           will be included in the returned files dict
    :type  include_env_in_files: bool
    :return: tuple of files dict and the loaded environment as a dict
    :rtype:  (dict, dict)
    """
    files: ty.Dict[str, str] = {}
    env: ty.Dict[str, ty.Dict] = {}
    is_object = env_path_is_object and env_path_is_object(env_path)
    if is_object:
        raw_env = object_request and object_request('GET', env_path)
        env = environment_format.parse(raw_env)
        env_base_url = utils.base_url_for_url(env_path)
        resolve_environment_urls(env.get('resource_registry'), files, env_base_url, is_object=True, object_request=object_request)
    elif env_path:
        env_url = utils.normalise_file_path_to_url(env_path)
        env_base_url = utils.base_url_for_url(env_url)
        raw_env = request.urlopen(env_url).read()
        env = environment_format.parse(raw_env)
        resolve_environment_urls(env.get('resource_registry'), files, env_base_url)
        if include_env_in_files:
            files[env_url] = json.dumps(env)
    return (files, env)