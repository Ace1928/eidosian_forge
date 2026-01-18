import collections
from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import environment_format as env_fmt
from heat.common import exception
from heat.common.i18n import _
def merge_environments(environment_files, files, params, param_schemata):
    """Merges environment files into the stack input parameters.

    If a list of environment files have been specified, this call will
    pull the contents of each from the files dict, parse them as
    environments, and merge them into the stack input params. This
    behavior is the same as earlier versions of the Heat client that
    performed this params population client-side.

    :param environment_files: ordered names of the environment files
           found in the files dict
    :type  environment_files: list or None
    :param files: mapping of stack filenames to contents
    :type  files: dict
    :param params: parameters describing the stack
    :type  params: dict
    :param param_schemata: parameter schema dict
    :type  param_schemata: dict
    """
    if not environment_files:
        return
    available_strategies = {}
    for filename in environment_files:
        raw_env = files[filename]
        parsed_env = env_fmt.parse(raw_env)
        strategies_in_file = parsed_env.pop(env_fmt.PARAMETER_MERGE_STRATEGIES, {})
        for section_key, section_value in parsed_env.items():
            if section_value:
                if section_key in (env_fmt.PARAMETERS, env_fmt.PARAMETER_DEFAULTS):
                    params[section_key] = merge_parameters(params[section_key], section_value, param_schemata, strategies_in_file, available_strategies, filename)
                else:
                    params[section_key] = merge_map(params[section_key], section_value)