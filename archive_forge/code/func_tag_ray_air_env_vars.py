import collections
from enum import Enum
import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
def tag_ray_air_env_vars() -> bool:
    """Records usage of environment variables exposed by the Ray AIR libraries.

    NOTE: This does not track the values of the environment variables, nor
    does this track environment variables not explicitly included in the
    `all_ray_air_env_vars` allow-list.

    Returns:
        bool: True if at least one environment var is supplied by the user.
    """
    from ray.air.constants import AIR_ENV_VARS
    from ray.tune.constants import TUNE_ENV_VARS
    from ray.train.constants import TRAIN_ENV_VARS
    all_ray_air_env_vars = sorted(set().union(AIR_ENV_VARS, TUNE_ENV_VARS, TRAIN_ENV_VARS))
    user_supplied_env_vars = []
    for env_var in all_ray_air_env_vars:
        if env_var in os.environ:
            user_supplied_env_vars.append(env_var)
    if user_supplied_env_vars:
        env_vars_str = json.dumps(user_supplied_env_vars)
        record_extra_usage_tag(TagKey.AIR_ENV_VARS, env_vars_str)
        return True
    return False