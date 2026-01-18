import functools
import re
import uuid
import ovs.db.parser
import ovs.db.types
import ovs.json
import ovs.jsonrpc
import ovs.ovsuuid
import ovs.poller
import ovs.socket_util
from ovs.db import error
@staticmethod
def from_python(type_, value, row_to_uuid):
    """Returns a new Datum with the given ovs.db.types.Type 'type_'.  The
        new datum's value is taken from 'value', which must take the form
        described as a valid return value from Datum.to_python() for 'type'.

        Each scalar value within 'value' is initially passed through
        'row_to_uuid', which should convert objects that represent rows (if
        any) into uuid.UUID objects and return other data unchanged.

        Raises ovs.db.error.Error if 'value' is not in an appropriate form for
        'type_'."""
    d = {}
    if isinstance(value, dict):
        for k, v in value.items():
            ka = Atom.from_python(type_.key, row_to_uuid(k))
            va = Atom.from_python(type_.value, row_to_uuid(v))
            d[ka] = va
    elif isinstance(value, (list, set, tuple)):
        for k in value:
            ka = Atom.from_python(type_.key, row_to_uuid(k))
            d[ka] = None
    else:
        ka = Atom.from_python(type_.key, row_to_uuid(value))
        d[ka] = None
    datum = Datum(type_, d)
    datum.check_constraints()
    if not datum.conforms_to_type():
        raise error.Error('%d values when type requires between %d and %d' % (len(d), type_.n_min, type_.n_max))
    return datum