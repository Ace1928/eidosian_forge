from __future__ import annotations
from kombu.utils.compat import _detect_environment
from kombu.utils.imports import symbol_by_name
def get_transport_cls(transport: str | None=None) -> str | None:
    """Get transport class by name.

    The transport string is the full path to a transport class, e.g.::

        "kombu.transport.pyamqp:Transport"

    If the name does not include `"."` (is not fully qualified),
    the alias table will be consulted.
    """
    if transport not in _transport_cache:
        _transport_cache[transport] = resolve_transport(transport)
    return _transport_cache[transport]