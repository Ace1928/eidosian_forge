from __future__ import annotations
from kombu.utils.compat import _detect_environment
from kombu.utils.imports import symbol_by_name
def resolve_transport(transport: str | None=None) -> str | None:
    """Get transport by name.

    Arguments:
    ---------
        transport (Union[str, type]): This can be either
            an actual transport class, or the fully qualified
            path to a transport class, or the alias of a transport.
    """
    if isinstance(transport, str):
        try:
            transport = TRANSPORT_ALIASES[transport]
        except KeyError:
            if '.' not in transport and ':' not in transport:
                from kombu.utils.text import fmatch_best
                alt = fmatch_best(transport, TRANSPORT_ALIASES)
                if alt:
                    raise KeyError('No such transport: {}.  Did you mean {}?'.format(transport, alt))
                raise KeyError(f'No such transport: {transport}')
        else:
            if callable(transport):
                transport = transport()
        return symbol_by_name(transport)
    return transport