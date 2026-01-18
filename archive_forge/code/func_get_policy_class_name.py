import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def get_policy_class_name(policy_class: type):
    """Returns a string name for the provided policy class.

    Args:
        policy_class: RLlib policy class, e.g. A3CTorchPolicy, DQNTFPolicy, etc.

    Returns:
        A string name uniquely mapped to the given policy class.
    """
    name = re.sub('_traced$', '', policy_class.__name__)
    if name in POLICIES:
        return name
    return None