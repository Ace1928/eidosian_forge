from typing import TYPE_CHECKING, Any, Optional
def ep_matches(ep: EntryPoint, **params) -> bool:
    """
    Workaround for ``EntryPoint`` objects without the ``matches`` method.
    """
    try:
        return ep.matches(**params)
    except AttributeError:
        from . import EntryPoint
        return EntryPoint(ep.name, ep.value, ep.group).matches(**params)