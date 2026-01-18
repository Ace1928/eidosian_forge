from collections import deque
from typing import (
import h11
from .connection import Connection, ConnectionState, ConnectionType
from .events import AcceptConnection, Event, RejectConnection, RejectData, Request
from .extensions import Extension
from .typing import Headers
from .utilities import (
def server_extensions_handshake(requested: Iterable[str], supported: List[Extension]) -> Optional[bytes]:
    """Agree on the extensions to use returning an appropriate header value.

    This returns None if there are no agreed extensions
    """
    accepts: Dict[str, Union[bool, bytes]] = {}
    for offer in requested:
        name = offer.split(';', 1)[0].strip()
        for extension in supported:
            if extension.name == name:
                accept = extension.accept(offer)
                if isinstance(accept, bool):
                    if accept:
                        accepts[extension.name] = True
                elif accept is not None:
                    accepts[extension.name] = accept.encode('ascii')
    if accepts:
        extensions: List[bytes] = []
        for name, params in accepts.items():
            name_bytes = name.encode('ascii')
            if isinstance(params, bool):
                assert params
                extensions.append(name_bytes)
            elif params == b'':
                extensions.append(b'%s' % name_bytes)
            else:
                extensions.append(b'%s; %s' % (name_bytes, params))
        return b', '.join(extensions)
    return None