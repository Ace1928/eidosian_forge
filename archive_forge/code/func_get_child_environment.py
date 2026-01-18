import collections
import fnmatch
import glob
import itertools
import os.path
import re
import weakref
from oslo_config import cfg
from oslo_log import log
from heat.common import environment_format as env_fmt
from heat.common import exception
from heat.common.i18n import _
from heat.common import policy
from heat.engine import support
def get_child_environment(parent_env, child_params, item_to_remove=None, child_resource_name=None):
    """Build a child environment using the parent environment and params.

    This is built from the child_params and the parent env so some
    resources can use user-provided parameters as if they come from an
    environment.

    1. resource_registry must be merged (child env should be loaded after the
       parent env to take precedence).
    2. child parameters must overwrite the parent's as they won't be relevant
       in the child template.

    If `child_resource_name` is provided, resources in the registry will be
    replaced with the contents of the matching child resource plus anything
    that passes a wildcard match.
    """

    def is_flat_params(env_or_param):
        if env_or_param is None:
            return False
        for sect in env_fmt.SECTIONS:
            if sect in env_or_param:
                return False
        return True
    child_env = parent_env.user_env_as_dict()
    child_env[env_fmt.PARAMETERS] = {}
    flat_params = is_flat_params(child_params)
    new_env = Environment()
    if flat_params and child_params is not None:
        child_env[env_fmt.PARAMETERS] = child_params
    new_env.load(child_env)
    if not flat_params and child_params is not None:
        new_env.load(child_params)
    if item_to_remove is not None:
        new_env.registry.remove_item(item_to_remove)
    if child_resource_name:
        new_env.registry.remove_resources_except(child_resource_name)
    return new_env