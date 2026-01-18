import contextlib
from functools import partial
from ray.rllib.utils.annotations import override, PublicAPI, DeveloperAPI
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.filter import Filter
from ray.rllib.utils.filter_manager import FilterManager
from ray.rllib.utils.framework import (
from ray.rllib.utils.numpy import (
from ray.rllib.utils.pre_checks.env import check_env
from ray.rllib.utils.schedules import (
from ray.rllib.utils.test_utils import (
from ray.tune.utils import merge_dicts, deep_update
@DeveloperAPI
def add_mixins(base, mixins, reversed=False):
    """Returns a new class with mixins applied in priority order."""
    mixins = list(mixins or [])
    while mixins:
        if reversed:

            class new_base(base, mixins.pop()):
                pass
        else:

            class new_base(mixins.pop(), base):
                pass
        base = new_base
    return base