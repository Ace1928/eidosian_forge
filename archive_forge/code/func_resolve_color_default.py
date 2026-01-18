import typing as t
from threading import local
def resolve_color_default(color: t.Optional[bool]=None) -> t.Optional[bool]:
    """Internal helper to get the default value of the color flag.  If a
    value is passed it's returned unchanged, otherwise it's looked up from
    the current context.
    """
    if color is not None:
        return color
    ctx = get_current_context(silent=True)
    if ctx is not None:
        return ctx.color
    return None