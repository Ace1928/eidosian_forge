from typing import TYPE_CHECKING, Any, Optional
def normalized_name(dist: Distribution) -> Optional[str]:
    """
    Honor name normalization for distributions that don't provide ``_normalized_name``.
    """
    try:
        return dist._normalized_name
    except AttributeError:
        from . import Prepared
        return Prepared.normalize(getattr(dist, 'name', None) or dist.metadata['Name'])