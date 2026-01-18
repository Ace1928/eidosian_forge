from ovs.flow.decoders import (
from ovs.flow.kv import (
from ovs.flow.list import nested_list_decoder, ListDecoders
from ovs.flow.ofp_fields import field_decoders, field_aliases
def decode_load_field(value):
    """Decodes LOAD actions such as: 'load:value->dst'."""
    parts = value.split('->')
    if len(parts) != 2:
        raise ValueError('Malformed load action : %s' % value)
    try:
        return {'value': int(parts[0], 0), 'dst': decode_field(parts[1])}
    except ValueError:
        return {'src': decode_field(parts[0]), 'dst': decode_field(parts[1])}