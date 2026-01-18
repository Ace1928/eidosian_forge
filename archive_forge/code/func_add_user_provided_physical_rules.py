from typing import List
from ray.data._internal.logical.interfaces.optimizer import Rule
def add_user_provided_physical_rules(default_rules: List[Rule]) -> List[Rule]:
    """
    Users can provide extra physical optimization rules here
    to be used in `PhysicalOptimizer`.

    Args:
        default_rules: the default physical optimization rules.

    Returns:
        The final physical optimization rules to be used in `PhysicalOptimizer`.
    """
    return default_rules