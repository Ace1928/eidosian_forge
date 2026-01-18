from ovs.flow.decoders import (
from ovs.flow.kv import (
from ovs.flow.list import nested_list_decoder, ListDecoders
from ovs.flow.ofp_fields import field_decoders, field_aliases
def decode_zone(value):
    """Decodes the value of the 'zone' keyword (part of the ct action)."""
    try:
        return int(value, 0)
    except ValueError:
        pass
    return decode_field(value)