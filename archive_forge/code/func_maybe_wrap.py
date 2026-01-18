import dns.exception
from ._compat import binary_type, string_types, PY2
def maybe_wrap(wire):
    if isinstance(wire, WireData):
        return wire
    elif isinstance(wire, binary_type):
        return WireData(wire)
    elif isinstance(wire, string_types):
        return WireData(wire.encode())
    raise ValueError('unhandled type %s' % type(wire))