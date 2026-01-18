from ray.rllib.utils.deprecation import Deprecated
from ray.util.annotations import _mark_annotated
def is_overridden(obj):
    """Check whether a function has been overridden.

    Note, this only works for API calls decorated with OverrideToImplementCustomLogic
    or OverrideToImplementCustomLogic_CallToSuperRecommended.
    """
    return getattr(obj, '__is_overriden__', True)