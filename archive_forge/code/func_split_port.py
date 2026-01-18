import re
def split_port(port):
    if hasattr(port, 'legacy_repr'):
        port = port.legacy_repr()
    port = str(port)
    match = PORT_SPEC.match(port)
    if match is None:
        _raise_invalid_port(port)
    parts = match.groupdict()
    host = parts['host']
    proto = parts['proto'] or ''
    internal = port_range(parts['int'], parts['int_end'], proto)
    external = port_range(parts['ext'], parts['ext_end'], '', len(internal) == 1)
    if host is None:
        if external is not None and len(internal) != len(external):
            raise ValueError("Port ranges don't match in length")
        return (internal, external)
    else:
        if not external:
            external = [None] * len(internal)
        elif len(internal) != len(external):
            raise ValueError("Port ranges don't match in length")
        return (internal, [(host, ext_port) for ext_port in external])