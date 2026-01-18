import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def get_policy_class(name: str):
    """Return an actual policy class given the string name.

    Args:
        name: string name of the policy class.

    Returns:
        Actual policy class for the given name.
    """
    if name not in POLICIES:
        return None
    path = POLICIES[name]
    module = importlib.import_module('ray.rllib.algorithms.' + path)
    if not hasattr(module, name):
        return None
    return getattr(module, name)